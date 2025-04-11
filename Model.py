import numpy as np
import threading
import time
import pylsl  # LSL library
from pylsl import StreamInlet, resolve_stream
import sys
from queue import Queue  # Use the thread-safe Queue
from scipy.signal import filtfilt
from scipy.fft import fft, fftfreq
import pickle
from scipy import signal
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 1880)

class EEGModel:
    def __init__(self, num_channels=8, samp_freq=250, window_size_second=4, band_pass = (2,40),stream_name="SomSom", status_queue=None, command_queue=None):
        self.num_channels = num_channels
        self.window_size = samp_freq * window_size_second
        self.eeg_data = np.zeros((self.num_channels, self.window_size))  # Real-time data buffer
        self.main_channel_roll = np.zeros((self.num_channels, self.window_size))  # For rolling samples
        self.stream_name = stream_name  # Name of the LSL stream to connect to
        self.status_queue = status_queue  # Queue for updating connection status
        self.queue1 = Queue(maxsize=10000)  # For storing LSL data, limited to prevent overflow
        self.queue2 = Queue(maxsize=10000)  # For storing rolled EEG data
        self.queue3 = Queue(maxsize=10000)  # For storing filtered EEG
        self.queue4 = Queue(maxsize=10000)  # For storing FFT data
        self.queue5 =  Queue(maxsize=10000)
        self.command_queue = command_queue
        self.running = False
        self.samp_freq = samp_freq
        self.command = "Non"
        self.EEG_epoch = np.zeros((num_channels, samp_freq))
        nyq = 0.5 * samp_freq
        self.b, self.a = signal.butter(4,[band_pass[0]/nyq,band_pass[1]/nyq],'bandpass')

    def start_streaming(self):
        """Start the LSL data stream and other processing functions."""
        self.running = True

        # Start only essential threads to manage memory usage
        threading.Thread(target=self.data_from_lsl, daemon=True).start()
        threading.Thread(target=self.rolling_samples, daemon=True).start()
        threading.Thread(target=self.filtering_windowed_data, daemon=True).start()
        threading.Thread(target=self.fft_process, daemon=True).start()
        threading.Thread(target=self.send_to_unity, daemon=True).start()
        
    def stop_streaming(self):
        """Stop the LSL data stream."""
        self.running = False

    def data_from_lsl(self):
        """Receive data from an LSL stream and update the model in real-time."""
        while self.running:
            streams = resolve_stream()
            for stream in streams:
                if stream.name() == self.stream_name:
                    inlet = StreamInlet(stream)
                    if self.status_queue:
                        self.status_queue.put((True, self.stream_name))
                    while self.running:
                        sample, _ = inlet.pull_chunk()
                        if sample:
                            data = np.array(sample).T
                            if data.shape[0] == self.num_channels:
                                self.EEG_epoch = np.roll(self.EEG_epoch, -data.shape[1], axis=1)
                                self.EEG_epoch[:, -data.shape[1]:] = data * 10e3
                                if not self.queue1.full():
                                    self.queue1.put(self.EEG_epoch)  # Add data if queue1 has space

                        time.sleep(0.05)

    def rolling_samples(self):
        """Continuously roll and update the main EEG data buffer."""
        while self.running:
            if not self.queue1.empty():
                try:
                    data = self.queue1.get()  # Get data from queue1
                    shift = data.shape[1]
                    # Roll and replace data within fixed memory
                    self.main_channel_roll = np.roll(self.main_channel_roll, -shift, axis=1)
                    self.main_channel_roll[:, -shift:] = data
                    if not self.queue2.full():
                        self.queue2.put(self.main_channel_roll)
                except Exception as e:
                    print(f"Rolling samples error: {e}")

            time.sleep(0.05)

    def filtering_windowed_data(self):
        while self.running:
            if not self.queue2.empty():
                data = self.queue2.get()
                filtered_data = filtfilt(self.b, self.a, data)
                if not self.queue3.full():
                    self.queue3.put(filtered_data)

            time.sleep(0.05)

    def fft_process(self):
        """Compute FFT on filtered data at a limited frequency to manage memory usage."""
        while self.running:
            if not self.queue3.empty():
                filtered_data = self.queue3.get()
                if filtered_data.shape[1] >= self.window_size:
                    yf = fft(filtered_data) 
                    power_spectrum = np.abs(yf) ** 2
                    xf = fftfreq(self.window_size, 1 / self.samp_freq)[:self.window_size // 2]
                    if not self.queue4.full():
                        self.queue4.put(power_spectrum[:])  # Keep only up to 40Hz for efficiency

                     #####Additional code of your preprocess###########

                    power_spectrum = power_spectrum.reshape(1,power_spectrum.shape[0],power_spectrum.shape[1])
                    fft_test = np.stack([arr.flatten() for arr in power_spectrum])

                    with open("trained_model/LDA_model.pkl", "rb") as file:
                        svm_model = pickle.load(file)

                    try:
                        predict = svm_model.predict(fft_test.reshape(1,fft_test.shape[1]))
                        predict_prob = svm_model.predict_proba(fft_test.reshape(1,fft_test.shape[1]))

                        arg_max = np.argmax(predict_prob[0])
                        # print(arg_max, predict_prob[0][arg_max])

                        if predict_prob[0][arg_max] > 0.6:
                            if predict[0] == 10:
                                self.command = '6Hz'
                            elif predict[0] == 8:
                                self.command = '12Hz'
                            elif predict[0] == 4:
                                self.command = '24Hz'
                            else:
                                self.command = '30Hz'
                        else:
                            self.command = 'Non'
                            
                        self.command_queue.put(self.command)

                    except Exception as e:
                        print(f"Preprocess model error: {e}")
                        
            time.sleep(0.05)  # Reduce FFT frequency to conserve memory and processing


    def send_to_unity(self):
        while self.running:
            if not self.command_queue.empty():
                send = self.command_queue.get()
                sock.sendto(str.encode(send), serverAddressPort)
            time.sleep(0.2)
                

# import numpy as np
# import threading
# import time
# import pylsl  # LSL library
# from pylsl import StreamInlet, resolve_stream
# import logging
# import sys
# from queue import Queue  # Use the thread-safe Queue
# from scipy.signal import filtfilt
# from scipy.fft import fft, fftfreq

# class EEGModel:
#     def __init__(self, num_channels=8, samp_freq = 250, window_size_second= 4 ,stream_name="SomSom", status_queue=None, command_queue=None):
#         self.num_channels = num_channels
#         self.window_size = samp_freq * window_size_second
#         self.eeg_data = np.zeros((self.num_channels, self.window_size))  # Real-time data buffer
#         self.main_channel_roll = np.zeros((self.num_channels, self.window_size))  # For rolling samples
#         self.stream_name = stream_name  # Name of the LSL stream to connect to
#         self.status_queue = status_queue  # Queue for updating connection status
#         self.queue1 = Queue(maxsize=20000)  # For stroring pull from LSL
#         self.queue2 = Queue(maxsize=20000)  # 
#         self.queue3 = Queue(maxsize=20000)  # For stroring rolled EEG data
#         self.queue4 = Queue(maxsize=20000)  # For storing filtered EEG
#         self.queue5 = Queue(maxsize=20000)  # For storing FFT data
#         self.running = False
#         self.command_queue = command_queue
#         self.samp_freq = samp_freq

#     def start_streaming(self):
#         """Start the LSL data stream and other processing functions."""
#         self.running = True

#         # Pull --> Count shift --> rolling 
#         threading.Thread(target=self.data_from_lsl, daemon=True).start()
#         threading.Thread(target=self.count_samples_from_lsl, daemon=True).start()
#         threading.Thread(target=self.rolling_samples, daemon=True).start()

#         # Filter --> Command --> FFT
#         threading.Thread(target=self.filtering_windowed_data, daemon=True).start()
#         threading.Thread(target=self.command_from_lsl, daemon=True).start()
#         threading.Thread(target=self.fft_filtered_window, daemon=True).start()

#     def stop_streaming(self):
#         """Stop the LSL data stream."""
#         self.running = False

#     def data_from_lsl(self):
#         """Receive data from an LSL stream and update the model in real-time."""
#         self.connect_status = True
#         while self.running:
#             print("Searching for LSL streams...")
#             all_streams_name = resolve_stream()
#             if all_streams_name:
#                 print("Found LSL Stream")

#                 for one_stream in all_streams_name:
#                     print(f"Found stream: {one_stream.name()}")

#                     # Connect to the stream with the specified name
#                     if one_stream.name() == self.stream_name:
#                         inlet_chn_0 = StreamInlet(one_stream)
#                         if self.status_queue:
#                             self.status_queue.put((self.connect_status, self.stream_name))
#                         print(f"Connected to stream: {self.stream_name}")
#                         break

#             while self.running:
#                 self.sub_samples_chn_0, timestamps = inlet_chn_0.pull_chunk()

#                 if len(self.sub_samples_chn_0) == 0:
#                     # print(f"No data received in this chunk, skipping update.")
#                     continue

#                 # Extract data for each of the 8 channels
#                 channel_data = [
#                     [sublist[i] for sublist in self.sub_samples_chn_0]
#                     for i in range(self.num_channels)
#                 ]

#                 self.final_list = np.array(channel_data)  # Convert to numpy array

#                 # Check if the data shape is valid before updating the buffer
#                 if self.final_list.shape[0] == self.num_channels and self.final_list.shape[1] > 0:
#                     try:
#                         self.queue1.put(self.final_list, timeout=1)  # Use `put` to safely add to the queue
#                         # print(f"Data pushed to queue1 with shape: {self.final_list.shape}")
#                     except:
#                         print("Queue1 is full, skipping this data chunk.") # Shouldn't happen
#                 else:
#                     print(f"Received empty or malformed data. Shape: {self.final_list.shape}")

#                 sys.stdout.flush()
#                 time.sleep(0.1)  # Adjust the sleep time according to data rate

#     def count_samples_from_lsl(self):
#         """Accumulate samples from queue1 until reaching (window_size/2) samples."""
#         main_chunk = np.empty((self.num_channels, 0))  # Initialize main_chunk
#         # print("Starting count_samples_from_lsl function...")

#         while self.running:
#             if not self.queue1.empty(): 
#                 try:
#                     # new_data = self.queue1.get(timeout=1)  # Get new data from queue1 with timeout for safety
#                     new_data = self.queue1.get() 
#                     main_chunk = np.hstack((main_chunk, new_data))  

#                     if main_chunk.shape[1] >= int(self.window_size/2):  # Check if main_chunk has window_size/2 samples
#                         # print(f"main_chunk reached 1000 samples, shape: {main_chunk.shape}")

#                         # Send the first 1000 samples to queue2
#                         self.queue2.put(main_chunk[:, :int(self.window_size/2)], timeout=1)
                        
#                         main_chunk = main_chunk[:, int(self.window_size/2):] # Keep the leftover samples in main_chunk
#                         # print(f"Sent 1000 samples to queue2; remaining shape: {main_chunk.shape}")
#                 except:
#                     print("Failed to get data from queue1.")

#             time.sleep(0.1)  # Small sleep to prevent busy-waiting

#     def rolling_samples(self):
#         """Continuously roll and update the main EEG data buffer."""
#         print("Starting rolling_samples function...")

#         while self.running:
#             if not self.queue2.empty():
#                 try:
#                     queue2_samples = self.queue2.get()  # Safely get data from queue2
#                     shift = queue2_samples.shape[1]
                    
#                     # Debug print to check if samples are correctly received
#                     # print(f"Received data in rolling_samples with shape: {queue2_samples.shape}")

#                     # Roll and append new data to the rolling buffer
#                     for i in range(self.main_channel_roll.shape[0]):  # loop over 8 rows
#                         self.main_channel_roll[i, :] = np.roll(self.main_channel_roll[i, :], shift)  # np.roll each row individually

#                     # Update the last `shift` columns with new data
#                     self.main_channel_roll[:, -shift:] = queue2_samples
#                     # print(f"Rolling data updated: shift={shift}, shape={self.main_channel_roll.shape}")

#                     self.queue3.put(self.main_channel_roll)
#                     # print("Put Queue3 Success")
#                 except:
#                     print("Failed to get data from queue2.")

#             time.sleep(1)  # Small sleep to prevent busy-waiting

#     def filtering_windowed_data(self):
#         # Alpha
#         # b =  np.array([ 5.61656229e-06,  0.00000000e+00, -2.24662491e-05,  0.00000000e+00,
#         #                 3.36993737e-05,  0.00000000e+00, -2.24662491e-05,  0.00000000e+00,
#         #                 5.61656229e-06])
#         # a =  np.array([  1.,          -7.50374777,  24.85978405, -47.48371343,  57.1853827,
#         #                 -44.46331966 , 21.79786977 , -6.16111674,   0.76887274])
        
#         # 2 - 40 Hz
#         b =  np.array([ 0.01937756,  0.,         -0.07751023,  0.,          0.11626534,  0.,
#         -0.07751023,  0.,          0.01937756])
#         a =  np.array([  1.,          -5.38768157,  12.76053185, -17.56050076,  15.5193986,
#         -9.05528257,   3.39134151,  -0.74121795,   0.07341328])

#         # Beta 
#         # b =  np.array([ 0.00126098,  0.,         -0.00504392,  0.,          0.00756587,  0.,
#         # -0.00504392,  0.,          0.00126098])
#         # a =  np.array([  1.,          -6.04348351,  16.67170206, -27.34091883,  29.11854592,
#         # -20.6136341,    9.47748845,  -2.59178076,   0.32413385])
        
#         while True:
#             if not self.queue3.empty():
#                 self.window_for_filter = self.queue3.get()
#                 self.window_for_filter = np.array(self.window_for_filter) # Convert to numpy
#                 self.window_filtered = filtfilt(b, a, self.window_for_filter)
#                 # print(f"Filtering Finish Shape {self.window_filtered.shape}")

#                 self.queue4.put(self.window_filtered)
#                 # print("Put Queue4 Success")
#                 sys.stdout.flush()

#     def fft_filtered_window(self):
#         while True:
#             if not self.queue3.empty():
#                 self.filtered_window = self.queue4.get()
#                 self.filtered_window = np.array(self.filtered_window) # Convert to numpy

#                 if self.filtered_window.shape[1] >= self.window_size:
#                     self.yf_filtered = fft(self.filtered_window)
#                     self.xf = fftfreq(self.window_size, 1/self.samp_freq)[:self.window_size//2]
#                     self.queue5.put(self.yf_filtered)
#                     # print("Queue5 Done")

#                 # print(f"Shape FFT of xf {self.xf[0:321]}") # 0 - 40 Hz only
#                 # print(f"Shape FFT of yf {self.yf_filtered[0:321]}") 

    # def command_from_lsl(self):
    #     """Simulate command generation."""
    #     while self.running:
    #         if not self.queue5.empty():
    #             data = self.queue5.get()
    #             data = data.reshape(1,self.queue5.get().shape[0],self.queue5.get().shape[1])
    #             fft_test = np.stack([arr.flatten() for arr in data])

    #             with open("trained_model/SVM_model.pkl", "rb") as file:
    #                 svm_model = pickle.load(file)

    #             predict = svm_model.predict(fft_test.reshape(1,fft_test.shape[1]))

    #             if predict[0] == 2:
    #                 self.command = '6Hz'
    #             elif predict[0] == 4:
    #                 self.command = '12Hz'
    #             elif predict[0] == 8:
    #                 self.command = '24Hz'
    #             else:
    #                 self.command = '30Hz'

    #             print("predict_command", self.command)
    #             self.command_queue.put(self.command)

    #         time.sleep(1)













# import dearpygui.dearpygui as dpg
# import time
# import numpy as np
# import threading

# class RealTimeView:
#     def __init__(self, model, ch_names, samp_freq = 512, window_size_second = 4):
#         self.model = model
#         self.queue1 = self.model.queue1
#         self.status_queue = self.model.status_queue  
#         self.command_queue = self.model.command_queue
#         self.queue_fft = self.model.queue5  # Change queue5 name to queue_fft
#         self.num_channels = len(ch_names)  
#         self.window_size = samp_freq * window_size_second
#         self.plot_array = np.zeros((self.num_channels, self.window_size))  
#         self.status_text = "Disconnected" 
#         self.stream_name_text = "None" 
#         self.command_text = "Waiting for Command..."  
#         self.fft_plot_data = np.zeros(321)  # Assume 321 frequency bins for FFT (0-40 Hz)

#     def setup_windows(self):
#         """Setup DearPyGUI windows and plots."""
#         dpg.create_context()

#         # EEG Status window
#         with dpg.window(label="EEG Status", tag="status_window"):
#             self.status_text_tag = dpg.add_text(f"Status: {self.status_text}")  
#             self.stream_name_text_tag = dpg.add_text(f"Stream name: {self.stream_name_text}") 

#         # EEG Streaming window for displaying the signals
#         with dpg.window(label="EEG Streaming", tag="streaming_window", height=700, width=700): # size for EEG Streaming window
#             for i in range(self.num_channels):
#                 with dpg.plot(height=80, width=600, no_menus=True):
#                     x_axis = dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True)  # X-axis
#                     y_axis = dpg.add_plot_axis(dpg.mvYAxis, label=f"Chn {i+1}", no_tick_labels=True)  # Y-axis
#                     setattr(self, f"plot_line{i+1}", dpg.add_line_series(list(range(self.window_size)), self.plot_array[i], parent=y_axis))

#         # Command Window for displaying commands from LSL
#         with dpg.window(label="Command Window", tag="command_window"):
#             self.command_text_tag = dpg.add_text(f"Command: {self.command_text}")

#             # Add three buttons to manually change the command
#             with dpg.group(horizontal=True):  # Horizontal alignment for buttons
#                 dpg.add_button(label="Left", callback=self.set_command_left)
#                 dpg.add_button(label="Right", callback=self.set_command_right)
#                 dpg.add_button(label="Rest", callback=self.set_command_rest)

#          # FFT Window for displaying FFT results
#         with dpg.window(label="FFT Plot", tag="fft_window", height=700, width=400):  
#             for i in range(self.num_channels):
#                 with dpg.plot(height=80, width=600):
#                     x_axis = dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True)  # X-axis for frequency
#                     y_axis = dpg.add_plot_axis(dpg.mvYAxis, label=f"FFT Chn {i+1}", no_tick_labels=True)  # Y-axis for amplitude
#                     setattr(self, f"fft_line_tag{i+1}", dpg.add_line_series(list(range(161)), self.fft_plot_data, parent=y_axis))
#                     dpg.set_axis_limits(x_axis, 0, 21) 
#                     dpg.add_text("20 Hz", parent=y_axis, pos=[550, 20])  # Adjust the position for alignment
#                     # dpg.set_axis_limits(y_axis, ymin=0, ymax=1)

#         # Setup the DearPyGUI viewport
#         dpg.create_viewport(title="Real-Time EEG Viewer", width=1200, height=600)
#         dpg.setup_dearpygui()
#         dpg.show_viewport()

#         dpg.show_item("status_window")
#         dpg.show_item("streaming_window")
#         dpg.show_item("command_window")
#         dpg.show_item("fft_window")

#     def set_command_left(self, sender, app_data):
#         """Callback for Left button to update the command to 'Left'."""
#         self.command_text = "Command: left"
#         dpg.set_value(self.command_text_tag, self.command_text)
#         if self.command_queue is not None:
#             self.command_queue.put("left")
#         print("Command manually set to: left")

#     def set_command_right(self, sender, app_data):
#         """Callback for Right button to update the command to 'Right'."""
#         self.command_text = "Command: right"
#         dpg.set_value(self.command_text_tag, self.command_text)
#         if self.command_queue is not None:
#             self.command_queue.put("right")
#         print("Command manually set to: right")

#     def set_command_rest(self, sender, app_data):
#         """Callback for Rest button to update the command to 'Rest'."""
#         self.command_text = "Command: rest"
#         dpg.set_value(self.command_text_tag, self.command_text)
#         if self.command_queue is not None:
#             self.command_queue.put("rest")
#         print("Command manually set to: rest")

#     def update_status_window(self):
#         """Update the status and stream name in the EEG Status window."""
#         while dpg.is_dearpygui_running():
#             if not self.status_queue.empty():
#                 self.connect_status, self.stream_name_text = self.status_queue.get()  # Retrieve status data
#                 self.status_text = "Connected" if self.connect_status else "Disconnected"
                
#                 # Update the status and stream name texts in the EEG Status window
#                 dpg.set_value(self.status_text_tag, f"Status: {self.status_text}")
#                 dpg.set_value(self.stream_name_text_tag, f"Stream name: {self.stream_name_text}")
#                 print(f"Updated status: {self.status_text}, Stream name: {self.stream_name_text}")

#             time.sleep(1) 

#     def update_command_window(self):
#         """Update the command text in the Command window."""
#         while dpg.is_dearpygui_running():
#             if not self.command_queue.empty():
#                 command = self.command_queue.get()  
#                 self.command_text = f"Command: {command}"
#                 dpg.set_value(self.command_text_tag, self.command_text)

#             time.sleep(1)

#     def update_fft_window(self):
#         """Update the FFT plot window with new FFT data."""
#         # Init x frequency
#         self.xf = [0., 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 
#                 1., 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2., 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3., 3.125, 3.25, 3.375, 3.5, 3.625, 3.75, 3.875, 4., 4.125, 4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5., 5.125, 5.25, 5.375, 5.5, 5.625, 5.75, 5.875, 6., 6.125, 6.25, 6.375, 6.5, 6.625, 6.75, 6.875, 7., 7.125, 7.25, 7.375, 7.5, 7.625, 7.75, 7.875, 8., 8.125, 8.25, 8.375, 8.5, 8.625, 8.75, 8.875, 9., 9.125, 9.25, 9.375, 9.5, 9.625, 9.75, 9.875, 10., 10.125, 10.25, 10.375, 10.5, 10.625, 10.75, 10.875, 11., 11.125, 11.25, 11.375, 11.5, 11.625, 11.75, 11.875, 12., 12.125, 12.25, 12.375, 12.5, 12.625, 12.75, 12.875, 13., 13.125, 13.25, 13.375, 13.5, 13.625, 13.75, 13.875, 14., 14.125, 14.25, 14.375, 14.5, 14.625, 14.75, 14.875, 15., 15.125, 15.25, 15.375, 15.5, 15.625, 15.75, 15.875, 16., 16.125, 16.25, 16.375, 16.5, 16.625, 16.75, 16.875, 17., 17.125, 17.25, 17.375, 17.5, 17.625, 17.75, 17.875, 18., 18.125, 18.25, 18.375, 18.5, 18.625, 18.75, 18.875, 19., 19.125, 19.25, 19.375, 19.5, 19.625, 19.75, 19.875, 20.]
#         while dpg.is_dearpygui_running():
#             if not self.queue_fft.empty():
#                 self.fft_data = self.queue_fft.get()
#                 self.fft_data = np.array(self.fft_data)  # Convert to numpy array

#                 if self.fft_data.shape[0] == self.num_channels and self.fft_data.shape[1] >= 161:
#                     # print("Received FFT data for all channels") 

#                     for i in range(self.num_channels):
#                         fft_plot_data = np.abs(self.fft_data[i, :161]) 
#                         dpg.set_value(getattr(self, f"fft_line_tag{i+1}"), [self.xf, fft_plot_data])
#                         ymin, ymax = np.min(fft_plot_data), np.max(fft_plot_data)
#                         y_axis = dpg.get_item_parent(getattr(self, f"fft_line_tag{i+1}"))  
#                         dpg.set_axis_limits(y_axis, ymin=ymin - 10, ymax=ymax + 10)  

#                     # print("Updated FFT Plot for all channels")

#             time.sleep(1)
    
#     def render_loop(self):
#         """GUI rendering loop using data from queue1."""
#         subpart_count = 100 

#         # Start separate threads to update status and command windows
#         threading.Thread(target=self.update_status_window, daemon=True).start()
#         threading.Thread(target=self.update_command_window, daemon=True).start()
#         threading.Thread(target=self.update_fft_window, daemon=True).start()

#         while dpg.is_dearpygui_running():
#             # Check if there's new data in queue1
#             if not self.queue1.empty():
#                 self.new_data = self.queue1.get() 
#                 self.new_data = np.array(self.new_data)

#                 # Check that the new data is correctly shaped
#                 if self.new_data.shape[0] == self.num_channels and self.new_data.shape[1] > 0:
#                     subparts = np.array_split(self.new_data, subpart_count, axis=1)  

#                     for subpart in subparts:
#                         shift = subpart.shape[1]
#                         if shift > 0:
#                             for i in range(self.plot_array.shape[0]):  # loop over each channel
#                                 self.plot_array[i, :] = np.roll(self.plot_array[i, :], -shift)  # Shift left
                                
#                             self.plot_array[:, -shift:] = subpart

#                             # Update each channel dynamically
#                             for i in range(self.num_channels):
#                                 dpg.set_value(getattr(self, f"plot_line{i+1}"), [list(range(self.window_size)), self.plot_array[i]])

#                                 ymin, ymax = np.min(self.plot_array[i]), np.max(self.plot_array[i])                                
#                                 y_axis = dpg.get_item_parent(getattr(self, f"plot_line{i+1}"))  # Get the Y-axis item
#                                 dpg.set_axis_limits(y_axis, ymin=ymin - 0.5, ymax=ymax + 0.5)  # Adjust for dynamic range with padding

#                             dpg.render_dearpygui_frame()
#                 else:
#                     print(f"Unexpected data shape received in queue1: {self.new_data.shape}")

#         dpg.destroy_context()
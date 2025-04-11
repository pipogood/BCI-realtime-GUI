from Model import EEGModel
from View import RealTimeView
from Controller import RealTimeController
import multiprocessing as mp
import time
import pickle

############### Biosemi Channels ################
# 1. Select all channels
# with open("datasets/biosemi_chans.pkl", "rb") as file:
#     ch_names = pickle.load(file)

# 2. Select target channels
ch_names = ['O1','Oz','PO3','POz','Pz']


num_channels = len(ch_names)
samp_freq = 512
window_size_second = 4

if __name__ == '__main__':
    mp.freeze_support()  # Ensure compatibility with multiprocessing on Windows

    # Use multiprocessing queues for status updates
    status_queue = mp.Queue()
    queue1 = mp.Queue()
    command_queue = mp.Queue()

    # Initialize the Model and View with LSL streaming
    model = EEGModel(num_channels=num_channels, samp_freq = samp_freq, window_size_second = window_size_second, band_pass = (2,40),
                     stream_name="SomSom",status_queue=status_queue, command_queue=command_queue)
    
    view = RealTimeView(model, ch_names, samp_freq=samp_freq, window_size_second=window_size_second)

    # Start the streaming process in the model
    model.start_streaming()

    # # Run the DearPyGUI rendering loop in the main thread
    view.setup_windows()  # Setup both windows
    view.render_loop()    # Start the rendering loop

    # Stop the model when exiting
    model.stop_streaming()

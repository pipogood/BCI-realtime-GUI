
import dearpygui.dearpygui as dpg
import time
import numpy as np
import threading

class RealTimeView:
    def __init__(self, model, ch_names, samp_freq=512, window_size_second=4):
        self.model = model
        self.queue1 = self.model.queue1
        self.status_queue = self.model.status_queue  
        self.command_queue = self.model.command_queue
        self.queue_fft = self.model.queue4  # Change queue5 name to queue_fft
        self.num_channels = len(ch_names)  
        self.ch_names = ch_names  # Store channel names
        self.window_size = samp_freq * window_size_second
        self.plot_array = np.zeros((self.num_channels, self.window_size))  
        self.selected_channels = list(range(4))  # Initially select the first 4 channels
        self.status_text = "Disconnected" 
        self.stream_name_text = "None" 
        self.command_text = "Waiting for Command..."  
        self.fft_plot_data = np.zeros(321)  # Assume 321 frequency bins for FFT (0-40 Hz)

    def setup_windows(self):
        """Setup DearPyGUI windows and plots."""
        dpg.create_context()

        # Channel Selection window
        with dpg.window(label="Channel Selection", tag="channel_selection_window"):
            dpg.add_text("Select up to 4 channels to plot:")
            self.checkboxes = []
            for i, ch_name in enumerate(self.ch_names):
                checkbox = dpg.add_checkbox(label=ch_name, default_value=(i in self.selected_channels), callback=self.update_selected_channels, user_data=i)
                self.checkboxes.append(checkbox)

        # EEG Status window
        with dpg.window(label="EEG Status", tag="status_window"):
            self.status_text_tag = dpg.add_text(f"Status: {self.status_text}")  
            self.stream_name_text_tag = dpg.add_text(f"Stream name: {self.stream_name_text}") 

        # EEG Streaming window for displaying the signals
        with dpg.window(label="EEG Streaming", tag="streaming_window", height=700, width=700):
            self.plot_lines = []
            for i in range(4):  # Display only 4 selected channels
                with dpg.plot(height=100, width=600, no_menus=True):
                    x_axis = dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True)
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label=f"Chn {i+1}", no_tick_labels=True)
                    plot_line = dpg.add_line_series(list(range(self.window_size)), self.plot_array[i], parent=y_axis)
                    self.plot_lines.append(plot_line)

        # Command Window for displaying commands from LSL
        with dpg.window(label="Command Window", tag="command_window"):
            self.command_text_tag = dpg.add_text(f"Command: {self.command_text}")

            # Add three buttons to manually change the command
            with dpg.group(horizontal=True):  # Horizontal alignment for buttons
                dpg.add_button(label="6Hz", callback=self.set_command_6Hz)
                dpg.add_button(label="12Hz", callback=self.set_command_12Hz)
                dpg.add_button(label="24Hz", callback=self.set_command_24Hz)
                dpg.add_button(label="30Hz", callback=self.set_command_30Hz)

        # FFT Window for displaying FFT results
        with dpg.window(label="FFT Plot", tag="fft_window", height=700, width=400):  
            self.fft_lines = []
            for i in range(4):  # Display only for 4 selected channels
                with dpg.plot(height=120, width=600):
                    # Add X-axis with custom ticks as labels
                    x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)")
                    dpg.set_axis_limits(x_axis, 0, 40)  # Set range to 0-40 Hz

                    # Manually set tick labels by adding text items at positions
                    for tick_value in range(0, 41, 5):  # Tick every 5 Hz from 0 to 40
                        dpg.add_text(f"{tick_value}", pos=[tick_value * 15, -15], parent=x_axis)  # Adjust position if needed

                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label=f"FFT Chn {i+1}", no_tick_labels=True)
                    fft_line = dpg.add_line_series(list(range(161)), self.fft_plot_data, parent=y_axis)
                    self.fft_lines.append(fft_line)
                    dpg.set_axis_limits(x_axis, 0, 40)
                    
        # Setup the DearPyGUI viewport
        dpg.create_viewport(title="Real-Time EEG Viewer", width=1200, height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        dpg.show_item("status_window")
        dpg.show_item("streaming_window")
        dpg.show_item("fft_window")
        dpg.show_item("command_window")
        dpg.show_item("channel_selection_window")

    def set_command_6Hz(self, sender, app_data):
        self.command_text = "Command: 6Hz"
        dpg.set_value(self.command_text_tag, self.command_text)
        if self.command_queue is not None:
            self.command_queue.put("6Hz")
        print("Command manually set to: 6Hz")

    def set_command_12Hz(self, sender, app_data):
        self.command_text = "Command: 12Hz"
        dpg.set_value(self.command_text_tag, self.command_text)
        if self.command_queue is not None:
            self.command_queue.put("12Hz")
        print("Command manually set to: 12Hz")

    def set_command_24Hz(self, sender, app_data):
        self.command_text = "Command: 24Hz"
        dpg.set_value(self.command_text_tag, self.command_text)
        if self.command_queue is not None:
            self.command_queue.put("24Hz")
        print("Command manually set to: 24Hz")

    def set_command_30Hz(self, sender, app_data):
        self.command_text = "Command: 30Hz"
        dpg.set_value(self.command_text_tag, self.command_text)
        if self.command_queue is not None:
            self.command_queue.put("30Hz")
        print("Command manually set to: 30Hz")

    def update_command_window(self):
        """Update the command text in the Command window."""
        while dpg.is_dearpygui_running():
            if not self.command_queue.empty():
                command = self.command_queue.get()  
                self.command_text = f"Command: {command}"
                dpg.set_value(self.command_text_tag, self.command_text)
            time.sleep(0.1)

    def update_status_window(self):
        """Update the status and stream name in the EEG Status window."""
        while dpg.is_dearpygui_running():
            if not self.status_queue.empty():
                self.connect_status, self.stream_name_text = self.status_queue.get()  # Retrieve status data
                self.status_text = "Connected" if self.connect_status else "Disconnected"
    
                # Update the status and stream name texts in the EEG Status window
                dpg.set_value(self.status_text_tag, f"Status: {self.status_text}")
                dpg.set_value(self.stream_name_text_tag, f"Stream name: {self.stream_name_text}")
                print(f"Updated status: {self.status_text}, Stream name: {self.stream_name_text}")

            time.sleep(0.1) 

    def update_selected_channels(self, sender, app_data, user_data):
        """Callback for channel selection checkboxes."""
        selected = [dpg.get_value(checkbox) for checkbox in self.checkboxes]
        self.selected_channels = [i for i, is_selected in enumerate(selected) if is_selected]
        
        # Limit to only 4 selected channels
        if len(self.selected_channels) > 4:
            dpg.set_value(sender, False)
            self.selected_channels = self.selected_channels[:4]

        print(f"Updated selected channels: {self.selected_channels}")

    def update_plots(self):
        """Update the displayed data in selected channel plots."""
        for plot_idx, ch_idx in enumerate(self.selected_channels):
            if plot_idx >= 4:  # Avoid exceeding the number of available plots
                break
            dpg.set_value(self.plot_lines[plot_idx], [list(range(self.window_size)), self.plot_array[ch_idx]])
            ymin, ymax = np.min(self.plot_array[ch_idx]), np.max(self.plot_array[ch_idx])
            y_axis = dpg.get_item_parent(self.plot_lines[plot_idx])
            dpg.set_axis_limits(y_axis, ymin - 0.5, ymax + 0.5)
            

    def update_fft_window(self):
        """Update the FFT plot window with new FFT data for the selected channels."""
        # Initialize x-axis for frequency bins
        self.xf = np.linspace(0, 40, 161)  # 0 to 20 Hz, 161 frequency bins
        while dpg.is_dearpygui_running():
            if not self.queue_fft.empty():
                # Get the latest FFT data from the queue
                self.fft_data = np.array(self.queue_fft.get())

                # Check if the FFT data shape is correct
                if self.fft_data.shape[0] == self.num_channels and self.fft_data.shape[1] >= 161:
                    # Loop through each of the selected channels, up to a maximum of 4
                    for plot_idx, ch_idx in enumerate(self.selected_channels):
                        if plot_idx >= 4:
                            break  # Limit to 4 displayed plots
                        # Extract FFT data for the selected channel
                        fft_plot_data = np.abs(self.fft_data[ch_idx, :])
                        # Update the corresponding FFT plot
                        dpg.set_value(self.fft_lines[plot_idx], [self.xf, fft_plot_data])
                        # Dynamically set Y-axis limits for better visualization
                        ymin, ymax = np.min(fft_plot_data), np.max(fft_plot_data)
                        y_axis = dpg.get_item_parent(self.fft_lines[plot_idx])
                        dpg.set_axis_limits(y_axis, ymin=ymin - 10, ymax=ymax + 10)

            time.sleep(0.05)


    def render_loop(self):
        """GUI rendering loop with data from queue1, with memory constraints."""
        subpart_count = 100 

        # Start separate threads to update status and command windows
        threading.Thread(target=self.update_status_window, daemon=True).start()
        threading.Thread(target=self.update_command_window, daemon=True).start()
        threading.Thread(target=self.update_fft_window, daemon=True).start()

        while dpg.is_dearpygui_running():
            if not self.queue1.empty():
                # Use only the latest data in the queue to avoid memory buildup
                while not self.queue1.empty():
                    self.new_data = np.array(self.queue1.get())

                # Validate new data shape
                if self.new_data.shape[0] == self.num_channels:
                    # Break down data into smaller subparts to reduce memory footprint
                    subparts = np.array_split(self.new_data, subpart_count, axis=1)

                    for subpart in subparts:
                        shift = subpart.shape[1]
                        if shift > 0:
                            # Roll and update `plot_array` with minimal allocation
                            for i in range(self.plot_array.shape[0]):
                                self.plot_array[i, :] = np.roll(self.plot_array[i, :], -shift)
                            
                            self.plot_array[:, -shift:] = subpart
                            self.update_plots()
                            dpg.render_dearpygui_frame()
                else:
                    print(f"Unexpected data shape received in queue1: {self.new_data.shape}")

        dpg.destroy_context()


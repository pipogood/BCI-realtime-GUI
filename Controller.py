class RealTimeController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def start(self):
        """Start the real-time EEG data stream and the GUI rendering."""
        self.model.start_streaming()
        self.view.setup_windows()

    def stop(self):
        """Stop both the EEG data stream and the GUI rendering."""
        self.model.stop_streaming()

import numpy as np
import pyabf
import config

class EEGProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.abf = pyabf.ABF(file_path)
        self.sample_rate = self.abf.dataRate
        self.data = self._load_data()
        self.time = self._create_time_array()
        self.fs = self.abf.dataRate

    def _load_data(self):
        data = {}
        for channel in range(self.abf.channelCount):
            self.abf.setSweep(sweepNumber=0, channel=channel)
            data[f"Channel_{channel}"] = self.abf.sweepY + channel * 25
        return data

    def _create_time_array(self):
        return np.arange(0, len(self.data["Channel_0"]) / self.sample_rate, 1 / self.sample_rate)

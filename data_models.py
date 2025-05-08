class DataContainer:
    def __init__(self, data, time, sample_rate, title):
        self.data = data
        self.time = time
        self.sample_rate = sample_rate
        self.title = title

class BehaviorEvent:
    def __init__(self, start_time, end_time, event_type, data, time, sample_rate):
        self.start = start_time
        self.end = end_time
        self.type = event_type
        self.data = data
        self.time = time
        self.sample_rate = sample_rate
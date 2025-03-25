import time 

class Timer:
    """A basic timer to time the time of functions and also the time of the whole program"""
    def __init__(self):
        self.start_time = time.perf_counter()
        self.last_time = self.start_time

    def elapsed(self):                                   #allows for two timing modes, one for overall and one for multiple increments between calls
        current_time = time.perf_counter()
        interval_time = current_time - self.last_time 
        total_time = current_time - self.start_time 
        self.last_time = current_time
        return interval_time, total_time
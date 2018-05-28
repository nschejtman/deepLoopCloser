import time
import numpy as np
import math


class ETAClock:
    def __init__(self, total):
        self._registered_times = []
        self._completed = 0
        self._start = None
        self.total = total

    def start(self):
        self._start = time.time()

    def lap(self):
        self._completed += 1
        self._registered_times.append(time.time())

    def get_eta(self):
        if not self._registered_times:
            return "??"
        else:
            remaining = self.total - self._completed
            avg = np.sum(self._registered_times) / self._completed
            millis = remaining * avg
            hs = math.floor(millis / 3600000)
            millis -= hs * 3600000
            mins = math.floor(millis / 60000)
            millis -= mins * 60000
            sec = math.floor(millis / 1000)
            return "%d hs %d min %d s" % (hs, mins, sec)

import math
import os

from pathlib2 import Path

"""
This class iterates through all the files that match the specified extension in the specified directory, in filename 
order. At each step of the iteration it returns a batch of the specified batch size of absolute file paths for each of 
the files in the batch. The iteration stops at the last available batch for the specified batch size, ignoring the 
remainder files. This means that for this BufferedReader to cover all matching files (given extension & directory) the 
length of these must be a multiple of the specified batch size.
"""


class BufferedReader:
    def __init__(self, directory, extension, batch_size):
        self.batch_size = batch_size
        self.idx = 0
        self.directory = str(Path(directory).resolve())
        all_files = os.listdir(self.directory)
        self.files = list(filter(lambda f: os.path.splitext(f)[1] == extension, all_files))
        self.files = list(map(lambda f: "%s/%s" % (self.directory, f), self.files))
        self.files.sort()
        self.breakIdx = math.floor(len(self.files) / self.batch_size) * self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.breakIdx:
            raise StopIteration()
        self.idx += self.batch_size
        return self.files[self.idx - self.batch_size: min(self.idx, len(self.files))]

    def __len__(self):
        return math.floor(len(self.files) / self.batch_size)

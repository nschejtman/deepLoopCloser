import os
import math


class BufferedFileReader:
    def __init__(self, directory, extension, batch_size):
        self.batch_size = batch_size
        self.idx = 0

        all_files = os.listdir(directory)
        self.files = list(filter(lambda f: os.path.splitext(f)[1] == extension, all_files))
        self.files = list(map(lambda f: directory + "/" + f, self.files))
        self.files.sort()

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += self.batch_size
        return self.files[self.idx - self.batch_size: min(self.idx, len(self.files) - 1)]

    def __len__(self):
        return math.ceil(len(self.files) / self.batch_size)

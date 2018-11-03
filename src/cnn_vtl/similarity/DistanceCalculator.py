import numpy as np


def _bitwise_diff(t: tuple):
    return bin(t[0] ^ t[1]).count('1')


class DistanceCalculator:
    @staticmethod
    def calculate_distance(desc1: list, desc2: list):
        diffs = map(_bitwise_diff, zip(desc1, desc2))
        return np.sum(list(diffs))

import numpy as np

from src.cnn_vtl.network.cnn_vtl import CnnVtl
from src.cnn_vtl.similarity.DistanceCalculator import DistanceCalculator

network = CnnVtl()
shape = network.input_shape
input_ = np.ones(shape=shape)
y1 = network.transform(input_)
y2 = network.transform(input_)
y2[0, 0] = 23
distance = DistanceCalculator.calculate_distance(y1[0], y2[0])
print(distance)

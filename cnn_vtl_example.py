from src.cnn_vtl.network.cnn_vtl import CnnVtl
from src.cnn_vtl.similarity.DistanceCalculator import DistanceCalculator
import numpy as np

network = CnnVtl()
shape = network.input_shape
shape2 = [1] + [i for i in shape[1:]]
input_ = np.ones(shape=shape2)
y1 = network.transform(input_)
y2 = network.transform(input_)
y2[0] = 23
distance = DistanceCalculator.calculate_distance(y1, y2)



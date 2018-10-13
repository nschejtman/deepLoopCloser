from model.cnn_vtl import CnnVtl
import numpy as np

net = CnnVtl()
a = net.transform(np.ones(shape=[1, 224, 224, 3]).tolist())
print(a)
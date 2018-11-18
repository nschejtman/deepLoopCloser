from src.cnn_vtl.network.cnn_vtl import CnnVtl
from src.cnn_vtl.similarity.DistanceCalculator import DistanceCalculator
import numpy as np
import os
import cv2
from tqdm import tqdm
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dataset_path = '/Users/nschejtman/projects/deepLoopCloser/datasets/outdoor_kennedylong'
files = os.listdir(dataset_path)
files.sort()
n_files = len(files)

logging.info('Creating network with shape=[%d, 192, 240, 3]' % n_files)
network = CnnVtl(input_shape=[n_files, 192, 240, 3])

logging.info('Reading files...')
dataset = list(map(lambda file: cv2.imread(dataset_path + "/" + file), files))
logging.info('Done reading files')

logging.info('Transforming images into descriptors...')
descriptors = network.transform(dataset)
logging.info('Done transforming images')

logging.info('Creating distance matrix...')
distance_matrix = np.full([n_files, n_files], -1)

for i in tqdm(range(n_files)):
    for j in range(n_files):
        distance = DistanceCalculator.calculate_distance(descriptors[i], descriptors[j])
        distance_matrix[i, j] = distance
logging.info('Creating distance matrix')


distance_img = 255 - distance_matrix / distance_matrix.max() * 255
cv2.imwrite("/Users/nschejtman/projects/deepLoopCloser/src/cnn_vtl/distance.png", distance_img)


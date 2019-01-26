import logging
import os
import sys
import warnings

import cv2
import numpy as np
from tqdm import tqdm

from src.sdav.input.CvInputParser import CvInputParser
from src.sdav.network.SDAV import SDAV
from src.sdav.similarity.SimilarityCalculator import SimilarityCalculator

sys.path.insert(0, '/Users/nschejtman/projects/deepLoopCloser')

# Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Create dataset
network = SDAV()

logging.info("Transforming parsed dataset into descriptors")
dataset_path = "/Users/nschejtman/projects/deepLoopCloser/datasets/test"
descriptors = network.transform_all("%s/*" % dataset_path)

logging.info("Calculating similarity")
n_files = len(os.listdir(dataset_path))
similarity_matrix = np.full([n_files, n_files], -1)
calculator = SimilarityCalculator(np.array(descriptors))

for i in tqdm(range(n_files)):
    for j in range(i + 1, n_files):
        similarity = calculator.similarity_score(descriptors[i], descriptors[j])
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity


# Move similarity value range to image pixels value range
move_factor = 0 - similarity_matrix.min()
divide_factor = similarity_matrix.max() + move_factor
normalized_matrix = (similarity_matrix + move_factor) / divide_factor
similarity_img = 255 * normalized_matrix

# logging.info("Writing similarity image")
cv2.imwrite("/Users/nschejtman/projects/deepLoopCloser/src/sdav/similarity.png", similarity_img)

# TODO add logging

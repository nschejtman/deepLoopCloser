from src.sdav.similarity.SimilarityCalculator import SimilarityCalculator
from src.sdav.network.SDAV import SDAV
from src.sdav.input.CvInputParser import CvInputParser
import cv2
import os
import numpy as np
from tqdm import tqdm
import logging
import warnings
import sys

sys.path.insert(0, '/Users/nschejtman/projects/deepLoopCloser')

# Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Create dataset
dataset_path = '/Users/nschejtman/projects/deepLoopCloser/datasets/outdoor_kennedylong'
files = os.listdir(dataset_path)
files.sort()
n_files = len(files)

# Parse dataset
parser = CvInputParser()
logging.info("Parsing dataset")
dataset = list(map(lambda file: parser.parse_from_path(dataset_path + "/" + file), tqdm(files)))


# Convert with network
logging.info("Creating network")
network = SDAV()

logging.info("Transforming parsed dataset into descriptors")
descriptors = list(map(lambda frame: network.transform(frame), tqdm(dataset)))

logging.info("Calculating similarity")
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

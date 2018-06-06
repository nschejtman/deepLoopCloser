from InputParser import InputParser
from BufferedFileReader import BufferedFileReader
import cv2
import numpy as np
import os

n_keypoints = 30
patch_size = 40
# TODO make this file a class

dataset = "/Users/nicolas/projects/deepLoopCloser/Dataset/dataset_2"
reader = BufferedFileReader(dataset, ".ppm", 5)
parser = InputParser(n_keypoints, patch_size)
n_batches = len(reader)
for i, batch in enumerate(reader):
    print("Processing %d/%d" % (i + 1, n_batches))
    cv_batch = list(map(lambda file: cv2.imread(file, cv2.IMREAD_GRAYSCALE), batch))
    parsed_batch = parser.calculate_all(cv_batch)
    reshaped = np.array(parsed_batch).reshape(len(batch) * n_keypoints, patch_size ** 2)
    save_base = "%s_parsed" % dataset
    if not os.path.exists(save_base):
        os.mkdir(save_base)
    save_dir = "%s/[n=%d][p=%d]" % (save_base, n_keypoints, patch_size)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = "%s/[batches=%d]" % (save_dir, n_batches)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    n_digits = len(str(n_batches))
    np.savetxt("%s/[batch=%s].csv" % (save_dir, str(i).zfill(n_digits)), reshaped, delimiter=",")




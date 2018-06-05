from InputParser import InputParser
from BufferedFileReader import BufferedFileReader
import cv2
import numpy as np

n_keypoints = 30
patch_size = 40

dataset = "/Users/nicolas/projects/deepLoopCloser/Dataset/dataset_2"
reader = BufferedFileReader(dataset, ".ppm", 551)
parser = InputParser(n_keypoints, patch_size)

for i, batch in enumerate(reader):
    cv_batch = list(map(lambda file: cv2.imread(file, cv2.IMREAD_GRAYSCALE), batch))
    parsed_batch = parser.calculate_all(cv_batch, verbose=True)
    print("Reshape started")
    reshaped = np.array(parsed_batch).reshape(len(batch) * n_keypoints, patch_size ** 2)
    print("Reshape finished")
    print("Save started")
    np.savetxt(dataset + "[n=%d][p=%d].csv" % (n_keypoints, patch_size), reshaped, delimiter=",")
    print("Save finished")



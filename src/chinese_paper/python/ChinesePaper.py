from InputParser import InputParser
from DAVariant import DAVariant
from BufferedFileReader import BufferedFileReader
import cv2

n_keypoints = 30
patch_size = 40

reader = BufferedFileReader("/Users/nicolas/projects/deepLoopCloser/Dataset/dataset_1", ".ppm", 5)
parser = InputParser(n_keypoints, patch_size)
da = DAVariant(n_keypoints=n_keypoints, patch_size=patch_size)

for batch in reader:
    cv_batch = map(lambda file: cv2.imread(file, cv2.IMREAD_GRAYSCALE), batch)
    parsed_batch = parser.calculate_all(cv_batch)
    da.fit(parsed_batch)

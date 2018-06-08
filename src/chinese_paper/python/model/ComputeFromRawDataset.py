import cv2

from input.CvInputParser import InputParser
from model.DenoisingAutoencoderVariant import DAVariant
from utils.BufferedFileReader import BufferedFileReader

n_keypoints = 30
patch_size = 40

reader = BufferedFileReader("/Users/nicolas/projects/deepLoopCloser/Dataset/dataset_1", ".ppm", 5)
parser = InputParser(n_keypoints, patch_size)
da = DAVariant(n_keypoints=n_keypoints, patch_size=patch_size)

n_batches = len(reader)

for i, batch in enumerate(reader):
    print("Started batch: " + str(i) + "/" + str(n_batches))
    cv_batch = list(map(lambda file: cv2.imread(file, cv2.IMREAD_GRAYSCALE), batch))
    parsed_batch = parser.calculate_all(cv_batch)
    da.fit(parsed_batch, warm_start=True)
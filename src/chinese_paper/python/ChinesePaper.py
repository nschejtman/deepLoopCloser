from InputParser import InputParser
from DAVariant import DAVariant
from BufferedFileReader import BufferedFileReader
import cv2
from ETAClock import ETAClock

n_keypoints = 15  # 30
patch_size = 40

reader = BufferedFileReader("/Users/nicolas/projects/deepLoopCloser/Dataset/dataset_1", ".ppm", 5)
parser = InputParser(n_keypoints, patch_size)
da = DAVariant(n_keypoints=n_keypoints, patch_size=patch_size)

n_batches = len(reader)

clock = ETAClock(n_batches)
clock.start()
for i, batch in enumerate(reader):
    print("Started batch: " + str(i) + "/" + str(n_batches))
    print("ETA: " + str(clock.get_eta()))
    cv_batch = list(map(lambda file: cv2.imread(file, cv2.IMREAD_GRAYSCALE), batch))
    parsed_batch = parser.calculate_all(cv_batch)
    da.fit(parsed_batch, warm_start=True)
    clock.lap()


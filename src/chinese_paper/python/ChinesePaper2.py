from DAVariant import DAVariant
from ETAClock import ETAClock
import numpy as np
from BufferedFileReader import BufferedFileReader

dataset = "dataset_1"
n_keypoints = 30
patch_size = 40
n_batches = 111
batch_size = 5

dataset_format = "/Users/nicolas/projects/deepLoopCloser/Dataset/%s_parsed/[n=%d][p=%d]/[batches=%d]"
dataset_dir = dataset_format % (dataset, n_keypoints, patch_size, n_batches)

da = DAVariant(n_keypoints=n_keypoints, patch_size=patch_size)
reader = BufferedFileReader(dataset_dir, ".csv", 1)

clock = ETAClock(n_batches)
clock.start()
for i, batch in enumerate(reader):
    print("Started batch: " + str(i) + "/" + str(n_batches))
    print("ETA: " + str(clock.get_eta()))
    parsed_batch = np.genfromtxt(batch[0], delimiter=',')
    parsed_batch = parsed_batch.reshape(batch_size, n_keypoints, patch_size ** 2)
    da.fit(parsed_batch, warm_start=True, with_device_info=True)
    clock.lap()

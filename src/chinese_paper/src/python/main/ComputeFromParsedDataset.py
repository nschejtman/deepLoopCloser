import numpy as np

from model.DenoisingAutoencoderVariant import DAVariant
from utils.BufferedFileReader import BufferedReader

dataset = "outdoor_kennedylong"
n_keypoints = 30
patch_size = 40
n_batches = 118
batch_size = 9

dataset_format = "../../../../../Dataset/%s_parsed/[n=%d][p=%d]/[batches=%d]"
dataset_dir = dataset_format % (dataset, n_keypoints, patch_size, n_batches)

da = DAVariant(0, n_keypoints=n_keypoints, patch_size=patch_size, n_consecutive_frames=9)
reader = BufferedReader(dataset_dir, ".csv", 1)

for i, batch in enumerate(reader):
    print("Started batch: " + str(i) + "/" + str(n_batches))
    parsed_batch = np.genfromtxt(batch[0], delimiter=',')
    da.fit(parsed_batch, warm_start=True)

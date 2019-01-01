import numpy as np
import src.utils.TensorflowWrapper as tw
import tensorflow as tf
import sys

sys.path.insert(0, '/Users/nschejtman/projects/deepLoopCloser')


def test_example():
    x = tw.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    w = tw.constant([[2, 2], [2, 2]])
    expected = np.array([[[6, 6], [14, 14]], [[22, 22], [30, 30]], [[38, 38], [46, 46]]]).astype(np.float64)

    y = x.matmul(w.to_tf()).to_tf()

    with tf.Session() as sess:
        actual = sess.run(y)

    assert np.array_equal(expected, actual)


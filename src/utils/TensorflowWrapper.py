import tensorflow as tf


class TensorWrapper:
    def __init__(self, x):
        self.x = x

    def flat_batch(self):
        shape = self.shape()
        return self.reshape([shape[0] * shape[1], shape[2]])

    def batch(self, batch_size):
        shape = self.shape()
        return self.reshape([batch_size, tf.cast(shape[0] / batch_size, tf.int32), shape[1]])

    def batch_size(self):
        return self.shape()[0]

    def corrupt(self, mask):
        return TensorWrapper(tf.multiply(self.x, mask))

    def dimensions(self):
        return len(self.x.get_shape())

    def shape(self):
        return tf.shape(self.x)

    def reshape(self, shape):
        return TensorWrapper(tf.reshape(self.x, shape))

    def matmul(self, y):
        yw = TensorWrapper(y)
        y_dims = yw.dimensions()
        x_dims = self.dimensions()
        if x_dims == y_dims:
            return TensorWrapper(self.x @ y)
        else:
            # Broadcast
            batch_size = self.batch_size()
            return self.flat_batch().matmul(y).batch(batch_size)

    def add(self, y):
        return TensorWrapper(self.x + y)

    def sigmoid(self):
        return TensorWrapper(tf.nn.sigmoid(self.x))

    def to_tf(self):
        return self.x


def placeholder(dtype, shape):
    return TensorWrapper(tf.placeholder(dtype, shape=shape))


def constant(value, dtype=tf.float64):
    return TensorWrapper(tf.constant(value, dtype=dtype))

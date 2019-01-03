import tensorflow as tf


class TensorWrapper:
    def __init__(self, x):
        self.x = x

    def flat_batch(self):
        shape = self.shape()
        return self.reshape([shape[0] * shape[1], shape[2]])

    def batch(self, batch_size):
        batch_size = parameter_guard(batch_size)
        shape = self.shape()
        return self.reshape([batch_size, tf.cast(shape[0] / batch_size, tf.int32), shape[1]])

    def batch_size(self):
        if self.dimensions() == 3:  # TODO replace with rank
            return self.shape()[0]
        else:
            return 1

    def parameter_number(self):
        if self.dimensions() == 3:
            return self.shape()[1] * self.shape()[2]
        else:
            return self.shape()[0] * self.shape()[1]

    def corrupt(self, corruption_level):
        shape = self.shape()
        shape = shape[1:] if self.dimensions() == 3 else shape
        mask = random_mask(shape, corruption_level)
        return self.multiply(mask)

    def rank(self):
        return TensorWrapper(tf.rank(self.x))

    def dimensions(self):
        return len(self.x.get_shape())

    def shape(self):
        return TensorWrapper(tf.shape(self.x))

    def concat(self, y, axis=0):
        y = parameter_guard(y)
        return TensorWrapper(tf.concat([self.x, y], axis))

    def reshape(self, shape):
        shape = parameter_guard(shape)
        return TensorWrapper(tf.reshape(self.x, shape))

    def matmul(self, y):
        y = parameter_guard(y)
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
        y = parameter_guard(y)
        return TensorWrapper(self.x + y)

    def multiply(self, y):
        y = parameter_guard(y)
        return TensorWrapper(tf.multiply(self.x, y))

    def sigmoid(self):
        return TensorWrapper(tf.nn.sigmoid(self.x))

    def shuffle(self):
        return TensorWrapper(tf.random.shuffle(self.x))

    def __getitem__(self, item):
        item = parameter_guard(item)
        return TensorWrapper(self.x[item])

    def __mul__(self, y):
        y = parameter_guard(y)
        return TensorWrapper(self.x * y)

    def __rmul__(self, y):
        y = parameter_guard(y)
        return TensorWrapper(y * self.x)


    def __add__(self, y):
        y = parameter_guard(y)
        return TensorWrapper(self.x + y)

    def __sub__(self, y):
        y = parameter_guard(y)
        return TensorWrapper(self.x - y)

    def to_tf(self):
        return self.x


def parameter_guard(y):
    if isinstance(y, TensorWrapper):
        return y.to_tf()
    else:
        return y


def placeholder(dtype, shape):
    shape = parameter_guard(shape)
    return TensorWrapper(tf.placeholder(dtype, shape=shape))


def constant(value, shape=None, dtype=tf.float64):
    if shape is None:
        shape = []
    else:
        shape = parameter_guard(shape)
    return TensorWrapper(tf.constant(value, shape=shape, dtype=dtype))


def zeros(shape, dtype=tf.float64):
    shape = parameter_guard(shape)
    return TensorWrapper(tf.zeros(shape, dtype=dtype))


def ones(shape, dtype=tf.float64):
    shape = parameter_guard(shape)
    return TensorWrapper(tf.ones(shape, dtype=dtype))


def random_mask(shape, zeros_percentage, dtype=tf.float64):
    shape = parameter_guard(shape)
    zeros_percentage = parameter_guard(zeros_percentage)

    parameters = shape[0] * shape[1]
    n_zeros = parameters * zeros_percentage
    n_ones = parameters - n_zeros

    return ones(n_ones, dtype=dtype).concat(zeros(n_zeros, dtype=dtype)).shuffle()

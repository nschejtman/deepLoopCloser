import logging
import os

import numpy as np
import tensorflow as tf
from py_v8n import v8n

from src.utils.MathUtils import MathUtils
from src.utils.PathUtils import firstParentWithNamePath


class CnnVtl:
    def __init__(self, input_shape: list = (1, 224, 224, 3), batch_size: int = 10, compress_factor: float = 99.59):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.compress_factor = compress_factor
        self._define_model()

    def _validate_params(self):
        v8n().list_().length(3).every().int_().greater_than(0).validate(self.input_shape)
        v8n().int_().validate(self.batch_size)
        v8n().float_().between(0, 100).validate(self.compress_factor)

    def _define_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def _define_model(self):
        params = CnnVtl._get_pretrained_params_initializers()
        self.x = tf.placeholder(tf.float64, shape=self.input_shape, name='input')

        # Layer 1
        conv1 = tf.layers.conv2d(inputs=self.x,
                                 kernel_size=[11, 11],
                                 filters=96,
                                 strides=[4, 4],
                                 activation=tf.nn.relu,
                                 name='conv1',
                                 kernel_initializer=params['conv1']['weights'],
                                 bias_initializer=params['conv1']['biases'])

        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=[3, 3],
                                        strides=[2, 2],
                                        name='pool1')

        # Layer 2
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 kernel_size=[5, 5],
                                 filters=256,
                                 strides=[1, 1],
                                 activation=tf.nn.relu,
                                 padding='same',
                                 name='conv2',
                                 kernel_initializer=params['conv2']['weights'],
                                 bias_initializer=params['conv2']['biases'])

        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=[3, 3],
                                        strides=[2, 2],
                                        name='pool2')

        # Layer 3
        conv3 = tf.layers.conv2d(inputs=pool2,
                                 kernel_size=[3, 3],
                                 filters=384,
                                 strides=[1, 1],
                                 activation=tf.nn.relu,
                                 padding='same',
                                 name='conv3',
                                 kernel_initializer=params['conv3']['weights'],
                                 bias_initializer=params['conv3']['biases'])

        # Layer 4
        conv4 = tf.layers.conv2d(inputs=conv3,
                                 kernel_size=[3, 3],
                                 filters=384,
                                 strides=[1, 1],
                                 activation=tf.nn.relu,
                                 padding='same',
                                 name='conv4',
                                 kernel_initializer=params['conv4']['weights'],
                                 bias_initializer=params['conv4']['biases'])

        # Layer 5
        conv5 = tf.layers.conv2d(inputs=conv4,
                                 kernel_size=[3, 3],
                                 filters=256,
                                 strides=[1, 1],
                                 padding='same',
                                 name='conv5',
                                 kernel_initializer=params['conv5']['weights'],
                                 bias_initializer=params['conv5']['biases'])

        # Vectorize layers
        layers = [conv1, conv2, conv3, conv4, conv5]

        layer_sizes = list(map(lambda layer: np.prod(layer.shape.as_list()[1:]), layers))

        def vectorize(idx, layer):
            return tf.reshape(layer, [self.input_shape[0], layer_sizes[idx]])

        vectorized_layers = list(map(lambda tup: vectorize(tup[0], tup[1]), enumerate(layers)))

        # Merge descriptors
        d = tf.concat(vectorized_layers, axis=1)

        # Cast to 8 bit ints between [0, 255]
        max_ = tf.reduce_max(d, axis=1)
        min_ = tf.reduce_min(d, axis=1)

        max_reshaped = tf.reshape(max_, [self.input_shape[0], 1])
        min_reshaped = tf.reshape(min_, [self.input_shape[0], 1])

        d_scaled = (d - min_reshaped) * (tf.constant(255, dtype=tf.float64) / (max_reshaped - min_reshaped))
        d_int8 = tf.cast(d_scaled, dtype=tf.int8)

        # Compress descriptor
        mask = np.zeros((np.sum(layer_sizes)), dtype=bool)

        start = 0
        for i in layer_sizes:
            true_indices = np.random.choice(np.arange(start, start + i),
                                            size=MathUtils.compressed_size(i, self.compress_factor))
            start += i
            mask[true_indices] = True

        self.y = tf.boolean_mask(d_int8, mask, axis=1)

    def transform(self, x):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(self.y, feed_dict={self.x: x})

    # noinspection PyUnresolvedReferences
    # noinspection PyTypeChecker
    @staticmethod
    def _get_pretrained_params_initializers():
        root_path = firstParentWithNamePath(os.path.abspath(__file__), 'deepLoopCloser')
        layer_params = np.load(root_path + '/training/cnn_vtl/npy/bvlc_alexnet.npy', encoding='bytes').item()
        skip_layers = ['fc6', 'fc7', 'fc8']
        params = {}
        for layer_name in layer_params:
            if layer_name not in skip_layers:
                params[layer_name] = {
                    'weights': tf.constant_initializer(layer_params[layer_name][0]),
                    'biases': tf.constant_initializer(layer_params[layer_name][1])
                }
        return params

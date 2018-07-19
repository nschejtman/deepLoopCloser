import os

import numpy as np
import tensorflow as tf
from pathlib2 import Path

from input.InputGenerator import get_generator


class DA:
    def __init__(self, input_shape: list, hidden_units: int, sparse_level: float = 0.05, sparse_penalty: float = 1,
                 consecutive_penalty: float = 0.2, batch_size: int = 10, learning_rate: float = 0.1, epochs: int = 100, layer_n: int = 0):

        if len(input_shape) != 2:
            raise ValueError("Input shape must specify a R^2 matrix")

        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.sparse_level = sparse_level
        self.sparse_penalty = sparse_penalty
        self.consecutive_penalty = consecutive_penalty
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layer_n = layer_n

        self._sess = None

        self._define_model()
        self._define_loss()
        self._define_optimizer()
        self._define_saver()
        self._define_summaries()

    def _define_model(self):

        # Input
        self._x = tf.placeholder(tf.float64, shape=[self.batch_size, self.input_shape[0], self.input_shape[1]], name='input')
        self._x_flat = tf.reshape(self._x, shape=[self.batch_size * self.input_shape[0], self.input_shape[1]], name='flat_input')
        self._x_corrupted = self._corrupt_tensor(self._x_flat, name='corrupted_input')

        # Encoder
        self._w0 = tf.Variable(tf.random_normal([self.input_shape[1], self.hidden_units], dtype=tf.float64), name='encoder_weights')
        self._b0 = tf.Variable(tf.zeros([self.hidden_units], dtype=tf.float64), name='encoder_biases')
        self.hidden_response = tf.nn.sigmoid(self._x_corrupted @ self._w0 + self._b0, name='hidden_response')

        # Decoder
        self._w1 = tf.transpose(self._w0, name='decoder_weights')
        self._b1 = tf.Variable(tf.zeros(self.input_shape[1], dtype=tf.float64), name='decoder_biases')
        self._y = tf.nn.sigmoid(self.hidden_response @ self._w1 + self._b1, name='recovered_input')

    def _define_loss(self):

        # Average cross entropy
        cd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._x, logits=self._y), name='average_cross_entropy')

        # Sparsity constraint
        cs = tf.reduce_mean(tf.norm(self.hidden_response - self.sparse_level, axis=1, ord=1), name='sparsity_constraint')

        # Consecutive constraint
        hidden_response_batch = tf.reshape(self.hidden_response, [self.batch_size, self.input_shape[0], self.hidden_units])
        frames = tf.slice(hidden_response_batch, [0, 0, 0], [self.batch_size - 1, self.input_shape[0], self.hidden_units])
        frames_next = tf.slice(hidden_response_batch, [1, 0, 0], [self.batch_size - 1, self.input_shape[0], self.hidden_units])
        cc = tf.reduce_mean(tf.norm(frames - frames_next, axis=[1, 2], ord='euclidean'), axis=0, name="consecutive_constraint")

        # Loss
        self._loss = cd + self.sparse_penalty * cs + self.consecutive_penalty * cc

    def _define_optimizer(self):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_step = optimizer.minimize(self._loss)

    def _define_summaries(self):
        tf.summary.histogram("w0", self._w0)
        tf.summary.histogram("b0", self._b0)
        tf.summary.histogram("w1", self._w1)
        tf.summary.histogram("b1", self._b1)
        tf.summary.scalar("loss", self._loss)
        self.log_dir = './log'
        self._summary_op = tf.summary.merge_all()
        self._summary_writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())

    def _define_saver(self):
        self._saver = tf.train.Saver(save_relative_paths=True)
        self.save_dir = 'checkpoints/layer_%d_' % self.layer_n
        self.save_file = '%s/checkpoint_file' % self.save_dir

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def _load_or_init_session(self):
        if len(os.listdir(self.save_dir)) > 0:
            self._saver.restore(self._sess, tf.train.latest_checkpoint(self.save_dir))
        else:
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    def _create_dataset(self, file_pattern: str):
        generator = get_generator(file_pattern, self.input_shape)
        dataset = tf.data.Dataset.from_generator(generator, tf.float64)
        return dataset.batch(self.batch_size)

    @staticmethod
    def _corrupt_tensor(x: tf.Tensor, corruption_level: float = 0.3, name: str = None):
        shape = np.array(x.get_shape().as_list())
        n_elems = shape.prod()

        # Create the corruption mask
        zeros_mask = np.ones(n_elems)
        zeros_mask[:int(n_elems * corruption_level)] = 0
        np.random.shuffle(zeros_mask)

        ones_mask = (zeros_mask - 1) * (-1)
        random_mask = np.random.rand(n_elems) < 0.5
        ones_mask = ones_mask.astype(int) & random_mask.astype(int)

        zeros_mask = zeros_mask.reshape(shape)
        ones_mask = ones_mask.reshape(shape)

        # TF operations
        tf_zeros_mask = tf.constant(zeros_mask.astype(float))
        tf_ones_mask = tf.constant(ones_mask.astype(float))

        return tf.multiply(tf_zeros_mask, x, name=name) + tf_ones_mask

    def fit_dataset(self, file_pattern: str, verbose: bool = False):
        dataset = self._create_dataset(file_pattern)
        iterator = dataset.make_one_shot_iterator()

        with tf.Session() as self._sess:
            batch_n = 0
            while True:
                try:
                    self._load_or_init_session()

                    batch = iterator.get_next()
                    stacked_batch = tf.stack(batch)
                    x = self._sess.run(stacked_batch)

                    for step in range(self.epochs):
                        self._sess.run(self.train_step, feed_dict={self._x: x})

                        if step % 10 == 0:
                            summary_str = self._sess.run(self._summary_op, feed_dict={self._x: x})
                            self._summary_writer.add_summary(summary_str)

                        if verbose:
                            progress_str = "Batch: %d, Epoch: %d/%d, Loss: %s"

                            loss = self._sess.run(self._loss, feed_dict={self._x: x})
                            print(progress_str % (batch_n, step + 1, self.epochs, loss))

                    batch_n += 1
                    self._saver.save(self._sess, self.save_file)

                except tf.errors.OutOfRangeError:
                    break

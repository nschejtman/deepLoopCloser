# This hack allows to import the input module from a sibling directory
# noinspection PyUnboundLocalVariable
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname

    path.append(dirname(path[0]))
    __package__ = "main"
# End hack

import logging
import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from pathlib2 import Path

from input.InputGenerator import get_generator


class DA:
    def __init__(self,
                 input_shape: list,
                 hidden_units: int,
                 sparse_level: float = 0.05,
                 sparse_penalty: float = 1,
                 consecutive_penalty: float = 0.2,
                 batch_size: int = 10,
                 learning_rate: float = 0.1,
                 epochs: int = 100,
                 layer_n: int = 0,
                 corruption_level: float = 0.3):

        if len(input_shape) != 2:
            raise ValueError('Input shape must specify a rank-2 tensor')

        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.sparse_level = sparse_level
        self.sparse_penalty = sparse_penalty
        self.consecutive_penalty = consecutive_penalty
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.corruption_level = corruption_level
        self.layer_n = layer_n

        self._sess = None

        root_name = 'chinese_paper'
        script_path = os.path.abspath(__file__)
        self.train_path = script_path[:(script_path.find(root_name) + len(root_name))] + '/training'

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        self._define_model_variables()
        self._define_fitting_model()
        self._define_transforming_model()
        self._define_loss()
        self._define_optimizer()
        self._define_saver()
        self._define_summaries()

    def _define_model_variables(self):
        # Model variables are shared between fit and transform models
        with tf.name_scope('encoder_variables'):
            self._w0 = tf.Variable(tf.random_normal([self.input_shape[1], self.hidden_units], dtype=tf.float64), name='encoder_weights')
            self._b0 = tf.Variable(tf.zeros([self.hidden_units], dtype=tf.float64), name='encoder_biases')

        with tf.name_scope('decoder_variables'):
            self._w1 = tf.transpose(self._w0, name='decoder_weights')
            self._b1 = tf.Variable(tf.zeros(self.input_shape[1], dtype=tf.float64), name='decoder_biases')

    def _define_fitting_model(self):
        # Separate model for batch fitting
        with tf.name_scope('fitting'):
            batch_shape = [self.batch_size, self.input_shape[0], self.input_shape[1]]
            flat_batch_shape = [self.batch_size * self.input_shape[0], self.input_shape[1]]

            self._x_batch = tf.placeholder(tf.float64, shape=batch_shape, name='train_batch')
            self._x_flat = tf.reshape(self._x_batch, shape=flat_batch_shape, name='train_flat_batch')
            self._x_corrupted = self._corrupt_tensor(self._x_flat, name='train_corrupted_batch')

            self._h_batch = tf.nn.sigmoid(self._x_corrupted @ self._w0 + self._b0, name='train_hidden_response')
            self._y_batch = tf.nn.sigmoid(self._h_batch @ self._w1 + self._b1, name='recovered_input')

    def _define_transforming_model(self):
        with tf.name_scope('transforming'):
            self._x_single = tf.placeholder(tf.float64, shape=self.input_shape, name='input')
            self._h_single = tf.nn.sigmoid(self._x_single @ self._w0 + self._b0)

    def _define_loss(self):
        with tf.name_scope('loss'):
            with tf.name_scope('average_cross_entropy'):
                cd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._x_batch, logits=self._y_batch),
                                    name='average_cross_entropy')

            with tf.name_scope('sparsity_constraint'):
                cs = tf.reduce_mean(tf.norm(self._h_batch - self.sparse_level, axis=1, ord=1), name='sparsity_constraint')

            with tf.name_scope('consecutive_constraint'):
                hidden_response_batch = tf.reshape(self._h_batch, [self.batch_size, self.input_shape[0], self.hidden_units])
                frames = tf.slice(hidden_response_batch, [0, 0, 0], [self.batch_size - 1, self.input_shape[0], self.hidden_units])
                frames_next = tf.slice(hidden_response_batch, [1, 0, 0], [self.batch_size - 1, self.input_shape[0], self.hidden_units])
                cc = tf.reduce_mean(tf.norm(frames - frames_next, axis=[1, 2], ord='euclidean'), axis=0, name='consecutive_constraint')

            # Loss
            self._loss = cd + self.sparse_penalty * cs + self.consecutive_penalty * cc

    def _define_optimizer(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_step = optimizer.minimize(self._loss, global_step=self.global_step)

    def _define_summaries(self):
        tf.summary.histogram('w0', self._w0)
        tf.summary.histogram('b0', self._b0)
        tf.summary.histogram('w1', self._w1)
        tf.summary.histogram('b1', self._b1)
        tf.summary.scalar('loss', self._loss)
        self.log_dir = '%s/log' % self.train_path
        self._summary_op = tf.summary.merge_all()
        self._summary_writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())

    def _define_saver(self):
        with tf.name_scope('saver'):
            self._saver = tf.train.Saver(save_relative_paths=True)
            self.save_dir = '%s/checkpoints/layer_%d_' % (self.train_path, self.layer_n)
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
        return dataset.batch(self.batch_size).prefetch(self.batch_size)

    def _corrupt_tensor(self, x: tf.Tensor, name: str = None):
        shape = np.array(x.get_shape().as_list())
        n_elems = shape.prod()

        # Create the corruption mask
        zeros_mask = np.ones(n_elems)
        zeros_mask[:int(n_elems * self.corruption_level)] = 0
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

    def fit_dataset(self, file_pattern: str):
        dataset = self._create_dataset(file_pattern)
        iterator = dataset.make_one_shot_iterator()

        with tf.Session() as self._sess:
            batch_n = 0
            while True:
                try:
                    self._load_or_init_session()

                    batch = iterator.get_next()
                    stack_batch_op = tf.stack(batch)
                    stacked_batch = self._sess.run(stack_batch_op)

                    # if batch size is different from the specified batch size the fitting model won't work due to mismatching shapes
                    if len(stacked_batch) != self.batch_size:
                        logging.warning("Ignored last batch because it was smaller than the specified batch size. To avoid this choose "
                                        "a batch size that is a factor of the dataset size.")
                        break

                    for step in range(self.epochs):
                        self._sess.run(self.train_step, feed_dict={self._x_batch: stacked_batch})

                        if step + 1 % 10 == 0:
                            self._write_summaries(stacked_batch)
                            self._saver.save(self._sess, self.save_file, global_step=self.global_step)

                        self._log_progress(batch_n, step, stacked_batch)

                except tf.errors.OutOfRangeError:
                    break

    def _log_progress(self, batch_n, step, x_batch):
        progress_str = 'Batch: %d, Epoch: %d/%d, Loss: %s'
        loss = self._sess.run(self._loss, feed_dict={self._x_batch: x_batch})
        logging.info(progress_str % (batch_n, step + 1, self.epochs, loss))

    def _write_summaries(self, x_batch):
        summary_str = self._sess.run(self._summary_op, feed_dict={self._x_batch: x_batch})
        self._summary_writer.add_summary(summary_str)

    def transform(self, x):
        with tf.Session() as self._sess:
            self._load_or_init_session()
            return self._sess.run(self._h_single, feed_dict={self._x_single: x})


def add_arguments(arg_parser):
    # Positional arguments
    arg_parser.add_argument('operation', choices=['train', 'transform'], help='Operation to perform')
    # Named arguments
    arg_parser.add_argument('--dataset_dir', help='Path to the dataset directory', required=True)
    arg_parser.add_argument('--dataset_ext', help='Extension of the image files in the dataset directory', required=True)
    # Named optional arguments
    arg_parser.add_argument('--input_shape', help='Shape of the input layer', type=int, nargs=2, default=[30, 1681])
    arg_parser.add_argument('--hidden_units', help='Number of hidden units', type=int, default=2500)
    arg_parser.add_argument('--batch_size', help='Batch size for training', type=int, default=10)
    arg_parser.add_argument('--corruption_level', help='Percentage of input vector to corrupt', type=float, default=0.3)
    arg_parser.add_argument('--sparse_penalty', help='Penalty weight for the sparsity constraint', type=float, default=1)
    arg_parser.add_argument('--sparse_level', help='Threshold factor for the sparsity constraint', type=float, default=0.05)
    arg_parser.add_argument('--consecutive_penalty', help='Penalty weight for consecutive constraint', type=float, default=0.2)
    arg_parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.1)
    arg_parser.add_argument('--epochs', help='Number of epochs to train each batch', type=int, default=100)
    arg_parser.add_argument('--verbose', help='Verbosity level for operations', type=bool, default=True)


def main():
    # Get arguments
    parser = ArgumentParser(description='Use this main file to train the model')
    add_arguments(parser)
    conf = parser.parse_args()

    # Create model
    model = DA(conf.input_shape,
               conf.hidden_units,
               sparse_level=conf.sparse_level,
               sparse_penalty=conf.sparse_penalty,
               consecutive_penalty=conf.consecutive_penalty,
               batch_size=conf.batch_size,
               learning_rate=conf.learning_rate,
               epochs=conf.epochs,
               corruption_level=conf.corruption_level)

    dataset_path = ('%s/*.%s' % (conf.dataset_dir, conf.dataset_ext)).replace('*..', '*.').replace('//', '/')

    if conf.operation == 'train':
        model.fit_dataset(dataset_path)


if __name__ == '__main__':
    main()

import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import logging

from src.sdav.input.InputGenerator import get_generator
from src.utils.PathUtils import firstParentWithNamePath


class SDAV:
    def __init__(self):
        self._configure_logging()
        logging.info('Initializing sdav')

        self._set_train_path()
        self._define_model_and_training_params()

        self.losses = []
        self._sess = None

        self._define_model_variables()
        self._define_fitting_model()
        self._define_transforming_model()
        self._define_optimizer()
        self._define_saver()
        self._define_summaries()
        logging.info('Finished initializing sdav')

    def _define_model_and_training_params(self):
        self.input_shape = [30, 1681]
        self.hidden_units = [2500, 2500, 2500, 2500, 2500]
        self.batch_size = 10
        self.corruption_level = 0.3
        self.sparse_level: float = 0.05
        self.sparse_penalty: float = 1.0
        self.consecutive_penalty: float = 0.2
        self.batch_size: int = 10
        self.learning_rate: float = 0.1
        self.epochs: int = 100
        self.corruption_level: float = 0.3
        logging.info('Parameter configuration: {\n' +
                     '\tinput_shape: %s,\n' % self.input_shape +
                     '\thidden_units: %s,\n' % self.hidden_units +
                     '\tbatch_size: %s,\n' % self.batch_size +
                     '\tcorruption_level: %s,\n' % self.corruption_level +
                     '\tsparse_level: %s,\n' % self.sparse_level +
                     '\tsparse_penalty: %s,\n' % self.sparse_penalty +
                     '\tconsecutive_penalty: %s,\n' % self.consecutive_penalty +
                     '\tbatch_size: %s,\n' % self.batch_size +
                     '\tlearning_rate: %s,\n' % self.learning_rate +
                     '\tepochs: %s,\n' % self.epochs +
                     '\tcorruption_level: %s,\n' % self.corruption_level +
                     '}')

    def _define_summaries(self):
        logging.info('Defining summaries')
        tf.summary.histogram('w0-e', self._w0_e)
        tf.summary.histogram('b0-e', self._b0_e)
        tf.summary.histogram('w0-d', self._w0_d)
        tf.summary.histogram('b0-d', self._b0_d)

        tf.summary.histogram('w1-e', self._w1_e)
        tf.summary.histogram('b1-e', self._b1_e)
        tf.summary.histogram('w1-d', self._w1_d)
        tf.summary.histogram('b1-d', self._b1_d)

        tf.summary.histogram('w2-e', self._w2_e)
        tf.summary.histogram('b2-e', self._b2_e)
        tf.summary.histogram('w2-d', self._w2_d)
        tf.summary.histogram('b2-d', self._b2_d)

        tf.summary.histogram('w3-e', self._w3_e)
        tf.summary.histogram('b3-e', self._b3_e)
        tf.summary.histogram('w3-d', self._w3_d)
        tf.summary.histogram('b3-d', self._b3_d)

        tf.summary.histogram('w4-e', self._w4_e)
        tf.summary.histogram('b4-e', self._b4_e)
        tf.summary.histogram('w4-d', self._w4_d)
        tf.summary.histogram('b4-d', self._b4_d)

        for i, loss in enumerate(self.losses):
            tf.summary.scalar('loss_%d' % i, loss)

        self._summary_op = tf.summary.merge_all()
        self._summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())

    def _set_train_path(self):
        root_path = firstParentWithNamePath(os.path.abspath(__file__), 'deepLoopCloser')
        self.train_path = root_path + '/training/sdav'
        Path(self.train_path).mkdir(parents=True, exist_ok=True)

        self.checkpoints_path = self.train_path + '/checkpoints'
        Path(self.checkpoints_path).mkdir(parents=True, exist_ok=True)

        self.log_path = self.train_path + '/log'
        Path(self.log_path).mkdir(parents=True, exist_ok=True)

        logging.info('Set training path to %s' % self.train_path)

    def _write_summaries(self, x_batch):
        logging.info('Writing summaries')
        summary_str = self._sess.run(self._summary_op, feed_dict={self._x0: x_batch})
        self._summary_writer.add_summary(summary_str)

    def _configure_logging(self):
        logging.basicConfig(filename='deepLoopCloser.log', format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S')
        self.logger = logging.getLogger()
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)

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

    def _define_fitting_model(self):
        logging.info('Defining fitting network')
        batch_shape_0 = [self.batch_size, self.input_shape[0], self.input_shape[1]]
        flat_batch_shape_0 = [self.batch_size * self.input_shape[0], self.input_shape[1]]

        self._x0 = tf.placeholder(tf.float64, shape=batch_shape_0)
        self._x0_flat = tf.reshape(self._x0, shape=flat_batch_shape_0)
        self._x0_corrupted = self._corrupt_tensor(self._x0_flat)

        self._h0 = tf.nn.sigmoid(self._x0_corrupted @ self._w0_e + self._b0_e)
        self._y0 = tf.nn.sigmoid(self._h0 @ self._w0_d + self._b0_d)

        self._define_loss(self._x0, self._h0, self._y0, 0)

        batch_shape_1 = [self.batch_size, self.input_shape[0], self.hidden_units[0]]
        self._x1 = tf.reshape(self._h0, shape=batch_shape_1)
        self._x1_corrupted = self._corrupt_tensor(self._h0)
        self._h1 = tf.nn.sigmoid(self._x1_corrupted @ self._w1_e + self._b1_e)
        self._y1 = tf.nn.sigmoid(self._h1 @ self._w1_d + self._b1_d)

        self._define_loss(self._x1, self._h1, self._y1, 1)

        batch_shape_2 = [self.batch_size, self.input_shape[0], self.hidden_units[1]]
        self._x2 = tf.reshape(self._h1, shape=batch_shape_2)
        self._x2_corrupted = self._corrupt_tensor(self._h1)
        self._h2 = tf.nn.sigmoid(self._x2_corrupted @ self._w2_e + self._b2_e)
        self._y2 = tf.nn.sigmoid(self._h2 @ self._w2_d + self._b2_d)

        self._define_loss(self._x2, self._h2, self._y2, 2)

        batch_shape_3 = [self.batch_size, self.input_shape[0], self.hidden_units[2]]
        self._x3 = tf.reshape(self._h2, shape=batch_shape_3)
        self._x3_corrupted = self._corrupt_tensor(self._h2)
        self._h3 = tf.nn.sigmoid(self._x3_corrupted @ self._w3_e + self._b3_e)
        self._y3 = tf.nn.sigmoid(self._h3 @ self._w3_d + self._b3_d)

        self._define_loss(self._x3, self._h3, self._y3, 3)

        batch_shape_4 = [self.batch_size, self.input_shape[0], self.hidden_units[3]]
        self._x4 = tf.reshape(self._h3, shape=batch_shape_4)
        self._x4_corrupted = self._corrupt_tensor(self._h3)
        self._h4 = tf.nn.sigmoid(self._x4_corrupted @ self._w4_e + self._b4_e)
        self._y4 = tf.nn.sigmoid(self._h4 @ self._w4_d + self._b4_d)

        self._define_loss(self._x4, self._h4, self._y4, 4)

    def _define_transforming_model(self):
        self._x0_single = tf.placeholder(tf.float64, shape=self.input_shape)
        self._h0_single = tf.nn.sigmoid(self._x0_single @ self._w0_e + self._b0_e)
        self._h1_single = tf.nn.sigmoid(self._h0_single @ self._w1_e + self._b1_e)
        self._h2_single = tf.nn.sigmoid(self._h1_single @ self._w2_e + self._b2_e)
        self._h3_single = tf.nn.sigmoid(self._h2_single @ self._w3_e + self._b3_e)
        self._h4_single = tf.nn.sigmoid(self._h3_single @ self._w4_e + self._b4_e)

    def _define_loss(self, x, h, y, layer_n):
        logging.info('Defining loss for layer %d' % layer_n)
        cd_0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=x, logits=y), name='cd_%d' % layer_n)

        cs_0 = tf.reduce_mean(tf.norm(h - self.sparse_level, axis=1, ord=1), name='cs_%d' % layer_n)

        hidden_response_batch = tf.reshape(h, [self.batch_size, self.input_shape[0], self.hidden_units[layer_n]])
        frames = tf.slice(hidden_response_batch, [0, 0, 0], [self.batch_size - 1, self.input_shape[0],
                                                             self.hidden_units[0]])
        frames_next = tf.slice(hidden_response_batch, [1, 0, 0],
                               [self.batch_size - 1, self.input_shape[0], self.hidden_units[layer_n]])
        cc_0 = tf.reduce_mean(tf.norm(frames - frames_next, axis=[1, 2], ord='euclidean'), axis=0,
                              name='cc_%d' % layer_n)

        # Loss
        self.losses.append(cd_0 + self.sparse_penalty * cs_0 + self.consecutive_penalty * cc_0)

    def _define_model_variables(self):
        logging.info('Defining network variables')
        self._w0_e = tf.Variable(
            tf.random_normal([self.input_shape[1], self.hidden_units[0]], dtype=tf.float64))
        self._b0_e = tf.Variable(tf.zeros([self.hidden_units[0]], dtype=tf.float64))
        self._w0_d = tf.transpose(self._w0_e)
        self._b0_d = tf.Variable(tf.zeros(self.input_shape[1], dtype=tf.float64))

        self._w1_e = tf.Variable(
            tf.random_normal([self.hidden_units[0], self.hidden_units[1]], dtype=tf.float64))
        self._b1_e = tf.Variable(tf.zeros([self.hidden_units[1]], dtype=tf.float64))
        self._w1_d = tf.transpose(self._w1_e)
        self._b1_d = tf.Variable(tf.zeros(self.hidden_units[0], dtype=tf.float64))

        self._w2_e = tf.Variable(
            tf.random_normal([self.hidden_units[1], self.hidden_units[2]], dtype=tf.float64))
        self._b2_e = tf.Variable(tf.zeros([self.hidden_units[2]], dtype=tf.float64))
        self._w2_d = tf.transpose(self._w2_e)
        self._b2_d = tf.Variable(tf.zeros(self.hidden_units[1], dtype=tf.float64))

        self._w3_e = tf.Variable(
            tf.random_normal([self.hidden_units[2], self.hidden_units[3]], dtype=tf.float64))
        self._b3_e = tf.Variable(tf.zeros([self.hidden_units[3]], dtype=tf.float64))
        self._w3_d = tf.transpose(self._w3_e)
        self._b3_d = tf.Variable(tf.zeros(self.hidden_units[2], dtype=tf.float64))

        self._w4_e = tf.Variable(
            tf.random_normal([self.hidden_units[3], self.hidden_units[4]], dtype=tf.float64))
        self._b4_e = tf.Variable(tf.zeros([self.hidden_units[4]], dtype=tf.float64))
        self._w4_d = tf.transpose(self._w4_e)
        self._b4_d = tf.Variable(tf.zeros(self.hidden_units[3], dtype=tf.float64))

    def _create_dataset(self, file_pattern: str):
        logging.info('Creating dataset')
        generator = get_generator(file_pattern, self.input_shape)
        return tf.data.Dataset.from_generator(generator, tf.float64)

    def fit(self, file_pattern: str):
        dataset = self._create_dataset(file_pattern)
        return self.fit_dataset(dataset)

    def _define_optimizer(self):
        logging.info('Defining optimizer')
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_steps = list(map(lambda loss: optimizer.minimize(loss, global_step=self.global_step), self.losses))

    def _define_saver(self):
        logging.info('Defining saver')
        self._saver = tf.train.Saver(save_relative_paths=True)
        self.checkpoint_file = '%s/checkpoint_file' % self.checkpoints_path

    def _load_or_init_session(self):
        if len(os.listdir(self.checkpoints_path)) > 0:
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoints_path)
            logging.info('Restoring session from %s' % latest_checkpoint)
            self._saver.restore(self._sess, latest_checkpoint)
        else:
            init_op = tf.global_variables_initializer()
            logging.info('Initializing session')
            self._sess.run(init_op)

    def fit_dataset(self, dataset: tf.data.Dataset):
        dataset = dataset.batch(self.batch_size).prefetch(self.batch_size)
        iterator = dataset.make_initializable_iterator()
        with tf.Session() as self._sess:
            for i in range(5):
                self._sess.run(iterator.initializer)
                logging.info('Fitting layer %d' % i)
                batch_n = 0
                self._load_or_init_session()
                while True:
                    try:
                        batch = iterator.get_next()
                        stack_batch_op = tf.stack(batch)
                        stacked_batch = self._sess.run(stack_batch_op)

                        if len(stacked_batch) != self.batch_size:
                            break

                        for step in range(self.epochs):
                            self._sess.run(self.train_steps[i], feed_dict={self._x0: stacked_batch})
                            self._log_progress(batch_n, step, stacked_batch, i)

                        batch_n += 1

                        if (batch_n + 1) % 25 == 0:
                            self._write_summaries(stacked_batch)

                    except tf.errors.OutOfRangeError:
                        break

                logging.info(
                    'Saving trained params to %s with global_step %s' % (self.checkpoint_file, self.global_step))
                self._saver.save(self._sess, self.checkpoint_file, global_step=self.global_step)

    def _log_progress(self, batch_n, step, x_batch, layer_n):
        progress_str = '    Layer:%d Batch:%d fit, Epoch:%d/%d, Loss:%s'
        loss = self._sess.run(self.losses[layer_n], feed_dict={self._x0: x_batch})
        logging.info(progress_str % (layer_n, batch_n, step + 1, self.epochs, loss))

    def transform(self, frame):
        with tf.Session() as self._sess:
            self._load_or_init_session()

            return self._sess.run(self._h4_single, feed_dict={self._x0_single: frame})

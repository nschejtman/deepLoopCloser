import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import logging
import src.utils.TensorflowWrapper as tw
from src.sdav.input.InputGenerator import get_generator
from src.utils.PathUtils import firstParentWithNamePath


class SDAV:
    def __init__(self, verbosity=logging.WARNING):
        self._configure_logging(verbosity)

        self._set_train_path()
        self._define_params()

        self.losses = []
        self._sess = None

        self._define_model_variables()
        self._define_model()
        self._define_optimizer()
        self._define_saver()
        self._define_summaries()
        logging.info('Done initializing sdav')

    def _define_params(self):
        self.input_shape = [30, 1681]
        self.hidden_units = [2500, 2500, 2500, 2500, 2500]
        self.default_batch_size = 10
        self.sparse_level: float = 0.05
        self.sparse_penalty: float = 1.0
        self.consecutive_penalty: float = 0.2
        self.learning_rate: float = 0.1
        self.epochs: int = 100
        self.corruption_level: float = 0.3
        logging.info('Parameter configuration: {\n' +
                     '\tinput_shape: %s,\n' % self.input_shape +
                     '\thidden_units: %s,\n' % self.hidden_units +
                     '\default_batch_size: %s,\n' % self.default_batch_size +
                     '\tcorruption_level: %s,\n' % self.corruption_level +
                     '\tsparse_level: %s,\n' % self.sparse_level +
                     '\tsparse_penalty: %s,\n' % self.sparse_penalty +
                     '\tconsecutive_penalty: %s,\n' % self.consecutive_penalty +
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

    def _configure_logging(self, verbosity):
        logging.basicConfig(filename='deepLoopCloser.log', format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S')
        self.logger = logging.getLogger()
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(verbosity)

    @staticmethod
    def _get_corruption_mask(shape, corruption_level):
        n_elements = shape[0] * shape[1]
        n_corrupted_elements = round(n_elements * corruption_level)
        mask = np.ones(n_elements)
        mask[:n_corrupted_elements] = 0
        np.random.shuffle(mask)
        return mask.reshape(shape)

    def _define_model(self):
        logging.info('Defining network')
        # noinspection PyTypeChecker
        input_shape = [None] + self.input_shape
        self._corruption_level = tf.placeholder(tf.float64, shape=[])

        # Layer 0
        x0 = tw.placeholder(tf.float64, input_shape)
        batch_size = x0.batch_size()
        h0 = x0.corrupt(self._corruption_level).matmul(self._w0_e).add(self._b0_e).sigmoid()
        y0 = h0.matmul(self._w0_d).add(self._b0_d).sigmoid()
        self._define_loss_for_layer(x0.to_tf(), h0.to_tf(), y0.to_tf(), 0, batch_size.to_tf())

        # Layer 1
        l1_shape = [batch_size] + self.get_layer_input_shape(1)
        x1 = h0.reshape(l1_shape).corrupt(self._corruption_level).flat_batch()
        h1 = x1.matmul(self._w1_e).add(self._b1_e).sigmoid()
        y1 = h1.matmul(self._w1_d).add(self._b1_d).sigmoid()
        self._define_loss_for_layer(x1.to_tf(), h1.to_tf(), y1.to_tf(), 1, batch_size.to_tf())

        # Layer 2
        l2_shape = [batch_size] + self.get_layer_input_shape(2)
        x2 = h1.reshape(l2_shape).corrupt(self._corruption_level).flat_batch()
        h2 = x2.matmul(self._w2_e).add(self._b2_e).sigmoid()
        y2 = h2.matmul(self._w2_d).add(self._b2_d).sigmoid()
        self._define_loss_for_layer(x2.to_tf(), h2.to_tf(), y2.to_tf(), 2, batch_size.to_tf())

        # Layer 3
        l3_shape = [batch_size] + self.get_layer_input_shape(3)
        x3 = h2.reshape(l3_shape).corrupt(self._corruption_level).flat_batch()
        h3 = x3.matmul(self._w3_e).add(self._b3_e).sigmoid()
        y3 = h3.matmul(self._w3_d).add(self._b3_d).sigmoid()
        self._define_loss_for_layer(x3.to_tf(), h3.to_tf(), y3.to_tf(), 3, batch_size.to_tf())

        # Layer 4
        l4_shape = [batch_size] + self.get_layer_input_shape(4)
        x4 = h3.reshape(l4_shape).corrupt(self._corruption_level).flat_batch()
        h4 = x4.matmul(self._w4_e).add(self._b4_e).sigmoid()
        y4 = h4.matmul(self._w4_d).add(self._b4_d).sigmoid()
        self._define_loss_for_layer(x4.to_tf(), h4.to_tf(), y4.to_tf(), 4, batch_size.to_tf())

        # Expose input & output tensors
        self._x0 = x0.to_tf()
        self._h4 = h4.to_tf()

    def get_layer_input_shape(self, layer_n):
        if layer_n == 0:
            return self.input_shape
        else:
            return [self.input_shape[0], self.hidden_units[layer_n - 1]]

    def _define_loss_for_layer(self, x, h, y, layer_n, batch_size):
        cd_0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=x, logits=y), name='cd_%d' % layer_n)

        cs_0 = tf.reduce_mean(tf.norm(h - self.sparse_level, axis=1, ord=1), name='cs_%d' % layer_n)

        hidden_response_batch = tf.reshape(h,
                                           [batch_size, self.input_shape[0], self.hidden_units[layer_n]])
        frames = tf.slice(hidden_response_batch, [0, 0, 0], [batch_size - 1, self.input_shape[0],
                                                             self.hidden_units[0]])
        frames_next = tf.slice(hidden_response_batch, [1, 0, 0],
                               [batch_size - 1, self.input_shape[0], self.hidden_units[layer_n]])
        cc_0 = tf.reduce_mean(tf.norm(frames - frames_next, axis=[1, 2], ord='euclidean'), axis=0,
                              name='cc_%d' % layer_n)

        # Loss
        self.losses.append(cd_0 + self.sparse_penalty * cs_0 + self.consecutive_penalty * cc_0)

    def _define_model_variables(self):
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

    def get_dataset(self, file_pattern: str):
        generator = get_generator(file_pattern, self.input_shape)
        return tf.data.Dataset.from_generator(generator, tf.float64)

    def _define_optimizer(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_steps = list(map(lambda loss: optimizer.minimize(loss, global_step=self.global_step), self.losses))

    def _define_saver(self):
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
        dataset = dataset.batch(self.default_batch_size).prefetch(self.default_batch_size)
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

                        for step in range(self.epochs):
                            feed_dict = {
                                self._x0: stacked_batch,
                                self._corruption_level: self.corruption_level
                            }
                            self._sess.run(self.train_steps[i], feed_dict=feed_dict)
                            self._log_progress(batch_n, step, stacked_batch, i)

                        batch_n += 1

                        if (batch_n + 1) % 25 == 0:
                            self._write_summaries(stacked_batch)

                    except tf.errors.OutOfRangeError:
                        break

                logging.info(
                    'Saving trained params to %s with global_step %s' % (self.checkpoint_file, self.global_step))
                self._saver.save(self._sess, self.checkpoint_file, global_step=self.global_step)

    def fit(self, x):
        with tf.Session() as self._sess:
            self._load_or_init_session()

            feed_dict = {
                self._x0: x,
                self._corruption_level: self.corruption_level
            }

            for i in range(5):
                for step in range(self.epochs):
                    self._sess.run(self.train_steps[i], feed_dict=feed_dict)

    def get_layers_input_shapes(self):
        return list(map(self.get_layer_input_shape, range(1, 6)))

    def transform(self, x):
        with tf.Session() as self._sess:
            self._load_or_init_session()

            feed_dict = {
                self._x0: x,
                self._corruption_level: 0
            }

            return self._sess.run(self._h4, feed_dict=feed_dict)

    def _log_progress(self, batch_n, step, x_batch, layer_n):
        progress_str = '    Layer:%d Batch:%d fit, Epoch:%d/%d, Loss:%s'
        loss = self._sess.run(self.losses[layer_n], feed_dict={self._x0: x_batch, self._corruption_level: self.corruption_level})
        logging.info(progress_str % (layer_n, batch_n, step + 1, self.epochs, loss))

    def transform_dataset(self, file_pattern: str):
        dataset = self.get_dataset(file_pattern)
        iterator = dataset.make_initializable_iterator()
        with tf.Session() as self._sess:
            self._sess.run(iterator.initializer)
            self._load_or_init_session()
            batch = iterator.get_next()
            stack_dataset_op = tf.stack(batch)
            stacked_dataset = self._sess.run(stack_dataset_op)
            return self._sess.run(self._h4, feed_dict={self._x0: stacked_dataset})

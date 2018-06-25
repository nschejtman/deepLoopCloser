import os
from pathlib2 import Path
import numpy as np
import tensorflow as tf


class DAVariant:
    def __init__(self, layer_n, n_keypoints=30, patch_size=40, n_consecutive_frames=5, hidden_layer_dimension=2500,
                 corruption_level=0.3, sparse_penalty=1, sparse_level=0.05, consecutive_penalty=0.2, learning_rate=0.1,
                 epochs=100):
        """
        Model parameters
        :param n_keypoints:
        :param patch_size:
        :param n_consecutive_frames:
        :param hidden_layer_dimension:
        :param corruption_level:
        :param sparse_penalty:
        :param sparse_level:
        :param consecutive_penalty:
        :param learning_rate:
        :param epochs:
        """
        self._set_parameters(layer_n, consecutive_penalty, corruption_level, hidden_layer_dimension, learning_rate,
                             n_consecutive_frames, epochs, n_keypoints, patch_size, sparse_level,
                             sparse_penalty)

        self._set_utilities()

        self._build_model()

    def _set_utilities(self):
        self.sess = None
        self.tf_saver = None
        self.summary_writer = None

        self.checkpoint_dir = str(Path("saved_tf_sessions/layer[%d]" % self.layer_n).resolve())
        self.checkpoint_file = "%s/checkpoint-file" % self.checkpoint_dir
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.log_path = "./log"
        Path(self.log_path).mkdir(parents=True, exist_ok=True)



    def _set_parameters(self, layer_n, consecutive_penalty, corruption_level, hidden_layer_dimension, learning_rate,
                        n_consecutive_frames, n_epochs, n_keypoints, patch_size, sparse_level, sparse_penalty):
        self.layer_n = layer_n
        self.n = n_keypoints
        self.s = patch_size
        self.nb = n_consecutive_frames
        self.nf = hidden_layer_dimension
        self.c = corruption_level
        self.beta = sparse_penalty
        self.sh = sparse_level
        self.gamma = consecutive_penalty
        self.eta = learning_rate
        self.n_epochs = n_epochs

    def _build_model(self):
        # Build the computation graph
        self.x_placeholder = tf.placeholder(tf.float64, shape=[self.nb * self.n, self.s ** 2])
        self.x_corr = self._corrupt(self.x_placeholder, self.c)
        self.w0 = tf.Variable(tf.random_normal([self.s ** 2, self.nf], dtype=tf.float64))
        b0 = tf.Variable(tf.zeros([self.nf], dtype=tf.float64))
        self.h = tf.nn.sigmoid(self.x_corr @ self.w0 + b0)
        w1 = tf.transpose(self.w0)
        b1 = tf.Variable(tf.zeros([self.s ** 2], dtype=tf.float64))
        y = tf.nn.sigmoid(self.h @ w1 + b1)

        # Build the loss function
        # Average Cross entropy
        cd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.x_placeholder, logits=y))
        # Sparsity constraint
        cs = tf.reduce_mean(tf.norm(self.h - self.sh, axis=1, ord=1))
        # Consecutive constraint
        frames_batch = tf.reshape(self.h, [self.nb, self.n, self.nf])
        frames_i = tf.slice(frames_batch, [0, 0, 0], [self.nb - 1, self.n, self.nf])
        frames_i_plus_1 = tf.slice(frames_batch, [1, 0, 0], [self.nb - 1, self.n, self.nf])
        norm = tf.norm(frames_i - frames_i_plus_1, axis=[1, 2], ord='euclidean')
        cc = tf.reduce_mean(norm, axis=0)
        self.loss = cd + self.beta * cs + self.gamma * cc

        # Add summary ops to collect data
        tf.summary.histogram("w0", self.w0)
        tf.summary.histogram("b0", b0)
        tf.summary.histogram("w1", w1)
        tf.summary.histogram("b1", b1)
        tf.summary.scalar("loss", self.loss)
        self.summary = tf.summary.merge_all()

    @staticmethod
    def _corrupt(x, corruption_level=0.0):
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

        return tf.multiply(tf_zeros_mask, x) + tf_ones_mask

    def fit(self, x, warm_start=False, with_device_info=False):
        if with_device_info:
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:
                self._init_model_and_utils(warm_start)
                self._train_model(x)
        else:
            with tf.Session() as self.sess:
                self._init_model_and_utils(warm_start)
                self._train_model(x)

    def _init_model_and_utils(self, warm_start):
        self.summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
        self.tf_saver = tf.train.Saver(save_relative_paths=True)

        if warm_start and len(os.listdir(self.checkpoint_dir)) > 0:
            self.tf_saver.restore(self.sess, self.checkpoint_file)
        else:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

    def _train_model(self, x):
        # Declare the optimizer
        optimizer = tf.train.GradientDescentOptimizer(self.eta)
        train_fn = optimizer.minimize(self.loss)
        print(self.sess.run(self.w0)[0, 0])
        for step in range(self.n_epochs):
            self.sess.run(train_fn, feed_dict={self.x_placeholder: x})

            # Write logs for each iteration
            if step % 10 == 0:
                summary_str = self.sess.run(self.summary, feed_dict={self.x_placeholder: x})
                self.summary_writer.add_summary(summary_str)

            progress_str = "Epoch: %d/%d Loss: %s"
            print(progress_str % (step + 1, self.n_epochs, self.sess.run(self.loss, feed_dict={self.x_placeholder: x})))
        print(self.sess.run(self.w0)[0, 0])
        self.summary_writer.close()
        self.checkpoint_file2 = self.tf_saver.save(self.sess, self.checkpoint_file)

    def transform(self, x):
        with self.sess as sess:
            sess.run(self.h, feed_dict=x)

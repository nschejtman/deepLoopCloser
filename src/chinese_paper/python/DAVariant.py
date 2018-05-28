import tensorflow as tf
import numpy as np
import os


def corrupt(x, corruption_level=0.0):
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


class DAVariant:
    def __init__(self, n_keypoints=30, patch_size=40, n_consecutive_frames=5, hidden_layer_dimension=2500,
                 corruption_level=0.3, sparse_penalty=1, sparse_level=0.05, consecutive_penalty=0.2, learning_rate=0.1,
                 n_epochs=100):
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
        :param n_epochs:
        """
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

        self.sess = None
        self.tf_saver = None

        self.save_path = "./saved_tf_sessions"
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        self._build_model()

    def _build_model(self):
        # Build the computation graph
        self.x_placeholder = tf.placeholder(tf.float64, shape=[self.nb, self.n, self.s ** 2])
        self.x_extended = tf.reshape(self.x_placeholder, [self.nb * self.n, self.s ** 2])  # Reshape to simplify graph
        self.x_corr = corrupt(self.x_extended, self.c)
        w0 = tf.Variable(tf.random_normal([self.s ** 2, self.nf], dtype=tf.float64))
        b0 = tf.Variable(tf.zeros([self.nf], dtype=tf.float64))
        self.h = tf.nn.sigmoid(self.x_corr @ w0 + b0)
        w1 = tf.transpose(w0)
        b1 = tf.Variable(tf.zeros([self.s ** 2], dtype=tf.float64))
        y = tf.nn.sigmoid(self.h @ w1 + b1)

        # Build the loss function
        # Average Cross entropy
        cd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.x_extended, logits=y))
        # Sparsity constraint
        cs = tf.reduce_mean(tf.norm(self.h - self.sh, axis=1, ord=1))
        # Consecutive constraint
        frames_batch = tf.reshape(self.h, [self.nb, self.n, self.nf])
        frames_i = tf.slice(frames_batch, [0, 0, 0], [self.nb - 1, self.n, self.nf])
        frames_i_plus_1 = tf.slice(frames_batch, [1, 0, 0], [self.nb - 1, self.n, self.nf])
        norm = tf.norm(frames_i - frames_i_plus_1, axis=[1, 2], ord='euclidean')
        cc = tf.reduce_mean(norm, axis=0)
        self.loss = cd + self.beta * cs + self.gamma * cc

    def fit(self, x, warm_start=False):
        with tf.Session() as self.sess:
            self._init_model_and_utils(warm_start)
            self._train_model(x)

    def _init_model_and_utils(self, warm_start):
        # self.tf_merged_summaries = tf.summary.merge_all() TODO
        init = tf.global_variables_initializer()
        self.tf_saver = tf.train.Saver()
        self.sess.run(init)

        if warm_start and os.path.exists(self.save_path + "/davariant"):
            self.tf_saver.restore(self.sess, self.save_path + "/davariant")
            # self.tf_summary_writer = tf.train.SummaryWriter(self.tf_summary_dir, self.tf_session.graph_def) TODO

    def _train_model(self, x):
        # Declare the optimizer
        optimizer = tf.train.GradientDescentOptimizer(self.eta)
        train_fn = optimizer.minimize(self.loss)
        for step in range(self.n_epochs):
            self.sess.run(train_fn, feed_dict={self.x_placeholder: x})
            progress_str = "Epoch: %d/%d Loss: %s"
            print(progress_str % (step + 1, self.n_epochs, self.sess.run(self.loss, feed_dict={self.x_placeholder: x})))
        self.tf_saver.save(self.sess, self.save_path + "/davariant")

    def transform(self, x):
        with self.sess as sess:
            sess.run(self.h, feed_dict=x)

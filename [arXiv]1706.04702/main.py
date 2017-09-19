import time, math
import tensorflow as tf
import numpy as np

from scipy.stats import multivariate_normal as normal

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow import random_normal_initializer as norm_init
from tensorflow import random_uniform_initializer as unif_init
from tensorflow import constant_initializer as const_init


class SolveAllenCahn(object):
    """The fully-connected neural network model."""
    def __init__(self, sess):
        self.sess = sess
        # PDE parameters
        self.d = 100
        self.T = 0.3
        # Algorithm parameters
        self.n_time = 20
        self.n_layer = 4
        self.n_neuron = [self.d, self.d + 10, self.d + 10, self.d]
        self.batch_size = 64
        self.valid_size = 256
        self.n_maxstep = 4000
        self.n_displaystep = 100
        self.learning_rate = 5e-4
        self.Yini = [0.3, 0.6]
        # Constants and Variables
        self.h = (self.T + 0.0) / self.n_time
        self.sqrth = math.sqrt(self.h)
        self.t_stamp = np.arange(0, self.n_time) * self.h
        self._extra_train_ops = []


    def train(self):
        self.start_time = time.time()
        self.display_time = self.t_bd + self.start_time
        # Training Operations
        self.global_step = tf.get_variable('global_step', [],
                                           initializer=tf.constant_initializer(1),
                                           trainable=False, dtype=tf.int32)
        trainable_vars = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_vars)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_vars),
                                             global_step=self.global_step)
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)
        self.loss_history = []
        self.init_history = []
        # Validation Config
        dW_valid, X_valid = self.sample_path(self.valid_size)
        feed_dict_valid = {self.dW: dW_valid,
                           self.X: X_valid,
                           self.is_training: False}
        # Initialization
        step = 1
        print("--------> call initializer")
        self.sess.run(tf.global_variables_initializer())
        temp_loss = self.sess.run(self.loss, feed_dict=feed_dict_valid)
        temp_init = self.Y0.eval()[0]
        self.loss_history.append(temp_loss)
        self.init_history.append(temp_init)
        self.call_display(0, temp_loss, temp_init)
        # Start Iteration
        for _ in range(self.n_maxstep + 1):
            step = self.sess.run(self.global_step)
            dW_train, X_train = self.sample_path(self.batch_size)
            self.sess.run(self.train_op, feed_dict={
                self.dW: dW_train, self.X: X_train, self.is_training: True
            })
            if step % self.n_displaystep == 0:
                temp_loss = self.sess.run(self.loss, feed_dict=feed_dict_valid)
                temp_init = self.Y0.eval()[0]
                self.loss_history.append(temp_loss)
                self.init_history.append(temp_init)
                self.call_display(step, temp_loss, temp_init)
            step += 1
        self.end_time = time.time()
        print("running time: %.3f s" % (self.end_time + self.display_time))


    def build(self):
        start_time = time.time()
        # stacking subnetworks
        self.dW = tf.placeholder(tf.float64, [None, self.d, self.n_time], name='dW')
        self.X = tf.placeholder(tf.float64, [None, self.d, self.n_time + 1], name='X')
        self.is_training = tf.placeholder(tf.bool)
        self.Y0 = tf.Variable(tf.random_uniform([1],
                            minval=self.Yini[0], maxval=self.Yini[1], dtype=tf.float64))
        self.Z0 = tf.Variable(tf.random_uniform([1, self.d],
                            minval=-0.1, maxval=0.1, dtype=tf.float64))
        self.allones = tf.ones(tf.stack([tf.shape(self.dW)[0], 1]), dtype=tf.float64)
        Y = self.allones * self.Y0
        Z = tf.matmul(self.allones, self.Z0)
        with tf.variable_scope('forward'):
            for t in range(0, self.n_time - 1):
                Y = Y - self.f_tf(self.t_stamp[t], self.X[:, :, t], Y, Z) * self.h
                Y = Y + tf.reduce_sum(Z * self.dW[:, :, t], 1, keep_dims=True)
                Z = self._one_time_net(self.X[:, :, t + 1], str(t + 1)) / self.d
            # terminal time
            Y = Y - self.f_tf(self.t_stamp[self.n_time - 1],
                              self.X[:, :, self.n_time - 1],
                              Y, Z) * self.h
            Y = Y + tf.reduce_sum(Z * self.dW[:, :, self.n_time - 1], 1, keep_dims=True)
            term_delta = Y - self.g_tf(self.T, self.X[:, :, self.n_time])
            self.clipped_delta = tf.clip_by_value(term_delta, -50.0, 50.0)
            self.loss = tf.reduce_mean(self.clipped_delta ** 2)
        self.t_bd = time.time() - start_time


    def call_display(self, step_, loss_, init_):
        print("step: %5u ===> loss= %.4e, Y0= %.4e, " % (step_, loss_, init_)
              + "runtime: %4u" % (time.time() + self.display_time))


    def sample_path(self, n_size):
        dW_ = np.zeros([n_size, self.d, self.n_time])
        X_ = np.zeros([n_size, self.d, self.n_time + 1])
        for i in range(self.n_time):
            dW_[:, :, i] = np.reshape(normal.rvs(mean=np.zeros(self.d), cov=1,
                                                 size=n_size) * self.sqrth,
                                      (n_size, self.d))
            X_[:, :, i + 1] = X_[:, :, i] + np.sqrt(2) * dW_[:, :, i]
        return dW_, X_


    def f_tf(self, t, X, Y, Z):
        # nonlinear term
        return Y - tf.pow(Y, 3)

    def g_tf(self, t, X):
        # terminal conditions
        return 0.5 / (1 + 0.2 * tf.reduce_sum(X ** 2, 1, keep_dims=True))

    def _one_time_net(self, x, name):
        with tf.variable_scope(name):
            x_norm = self._batch_norm(x, name='layer0_normal')
            layer1 = self._one_layer(x_norm, self.n_neuron[1], name='layer1')
            layer2 = self._one_layer(layer1, self.n_neuron[2], name='layer2')
            z = self._one_layer(layer2, self.n_neuron[3], name='final')
        return z

    def _batch_norm(self, x, name):
        """Batch Normalization"""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable('beta', params_shape, tf.float64,
                                   norm_init(0.0, stddev=0.1, dtype=tf.float64))
            gamma = tf.get_variable('gamma', params_shape, tf.float64,
                                    unif_init(0.1, 0.5, dtype=tf.float64))
            mv_mean = tf.get_variable('moving_mean', params_shape, tf.float64,
                                      const_init(0.0, tf.float64), trainable=False)
            mv_var = tf.get_variable('moving_variance', params_shape, tf.float64,
                                     const_init(1.0, tf.float64), trainable=False)
            # Training Ops
            mean, variance = tf.nn.moments(x, [0], name='moments')
            hoge = assign_moving_average(mv_mean, mean, 0.99)
            piyo = assign_moving_average(mv_var, variance, 0.99)
            self._extra_train_ops.extend([hoge, piyo])
            mean, variance = control_flow_ops.cond(self.is_training,
                                                   lambda : (mean, variance),
                                                   lambda : (mv_mean, mv_var))
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-6)
            y.set_shape(x.get_shape())
            return y


    def _one_layer(self, input_, out_size, f_activation=tf.nn.relu, std=5.0, name='linear'):
        with tf.variable_scope(name):
            shape = input_.get_shape().as_list()
            w = tf.get_variable('Matrix', [shape[1], out_size], tf.float64,
                                norm_init(stddev=std / np.sqrt(shape[1] + out_size)))
            hidden = tf.matmul(input_, w)
            hidden_bn = self._batch_norm(hidden, name='normal')
        if f_activation != None:
            return f_activation(hidden_bn)
        else: return hidden_bn

GLOBAL_RANDOM_SEED = 1

def main():
    tf.reset_default_graph()
    with tf.Session() as sess:
        tf.set_random_seed(GLOBAL_RANDOM_SEED)
        print("========Begin to solve Allen-Cahn Equation========")
        model = SolveAllenCahn(sess)
        model.build()
        model.train()
        output = np.zeros((len(model.init_history), 3))
        output[:, 0] = np.arange(len(model.init_history)) * model.n_displaystep
        output[:, 1] = model.loss_history
        output[:, 2] = model.init_history
        np.savetxt("./AllenCahn_d100.csv", output, fmt=['%d', '%.5e', '%.5e'],
                   delimiter=',', header='step, loss function, target value, runtime',
                   comments='')
        

if __name__ == '__main__':
    np.random.seed(GLOBAL_RANDOM_SEED)
    main()
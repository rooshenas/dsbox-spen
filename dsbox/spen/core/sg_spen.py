import tensorflow as tf
import numpy as np

import tflearn
import tflearn.initializations as tfi
from enum import Enum
import math
import random


class InfInit(Enum):
    Random_Initialization = 1
    GT_Initialization = 2
    Zero_Initialization = 3


class TrainingType(Enum):
    Value_Matching = 1
    SSVM = 2
    Rank_Based = 3
    End2End = 4
    CLL = 5


class SPEN:
    def __init__(self, config):
        self.config = config
        self.x = tf.placeholder(tf.float32, shape=[None, self.config.input_num], name="InputX")
        self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name="LearningRate")
        self.is_training = tf.placeholder(tf.float32, shape=[], name="IsTraining")
        self.dropout_ph = tf.placeholder(tf.float32, shape=[], name="Dropout")
        self.bs_ph = tf.placeholder(tf.int32, shape=[], name="BatchSize")
        self.embedding = None
        self.log_const = {}
        self.log_search = {}
        #

    def init(self):
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)
        self.saver = tf.train.Saver()

        return self

    def init_embedding(self, embedding):
        self.sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: embedding})
        return self

    def set_train_iter(self, iter):
        self.train_iter = iter

    def print_vars(self):
        for v in self.spen_variables():
            print(v)

    def spen_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen")

    def energy_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen/en")

    def energy_g_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen/en/en.g")

    def fnet_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen/fx")

    def get_l2_loss(self):
        loss = 0.0
        en_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for v in en_vars:
            loss += tf.nn.l2_loss(v)
        return loss

    def var_to_indicator(self, vd):
        size = np.shape(vd)
        cat = np.zeros((size[0], self.config.output_num, self.config.dimension))
        for i in range(size[0]):
            for j in range(self.config.output_num):
                k = vd[i, j]
                cat[i, j, int(k)] = 1.0
        return np.reshape(cat, (size[0], self.config.output_num, self.config.dimension))

    def indicator_to_var(self, ind):
        size = np.shape(ind)
        y_cat_indicator = np.reshape(ind, (size[0], self.config.output_num, self.config.dimension))
        y_m = np.argmax(y_cat_indicator, 2)
        return y_m

    def reduce_learning_rate(self, factor):
        self.config.learning_rate *= factor

    def get_energy(self, xinput=None, yinput=None, embedding=None, reuse=False):
        raise NotImplementedError

    def createOptimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)

    def get_prediction_net(self, input=None, xinput=None, reuse=False):
        raise NotImplementedError

    def get_feature_net(self, xinput, output_num, embedding=None, reuse=False):
        raise NotImplementedError


    def rank_based_training(self):
        self.margin_weight_ph = tf.placeholder(tf.float32, shape=[], name="Margin")
        self.inf_penalty_weight_ph = tf.placeholder(tf.float32, shape=[], name="InfPenalty")
        self.yp_h_ind = tf.placeholder(tf.float32,
                                       shape=[None, self.config.output_num * self.config.dimension],
                                       name="YP_H")

        self.yp_l_ind = tf.placeholder(tf.float32,
                                       shape=[None, self.config.output_num * self.config.dimension],
                                       name="YP_L")

        yp_ind_sm_h = tf.nn.softmax(tf.reshape(self.yp_h_ind, [-1, self.config.output_num, self.config.dimension]))
        self.yp_h = tf.reshape(yp_ind_sm_h, [-1, self.config.output_num * self.config.dimension])

        yp_ind_sm_l = tf.nn.softmax(tf.reshape(self.yp_l_ind, [-1, self.config.output_num, self.config.dimension]))
        self.yp_l = tf.reshape(yp_ind_sm_l, [-1, self.config.output_num * self.config.dimension])

        self.value_h = tf.placeholder(tf.float32, shape=[None])
        self.value_l = tf.placeholder(tf.float32, shape=[None])


        self.yh_penalty = self.inf_penalty_weight_ph * tf.reduce_sum(tf.square(self.yp_h_ind), 1)
        self.yl_penalty = self.inf_penalty_weight_ph * tf.reduce_sum(tf.square(self.yp_l_ind), 1)

        self.energy_yh_ = self.get_energy(xinput=self.x, yinput=self.yp_h, embedding=self.embedding,
                                          reuse=False)
        self.energy_yl_ = self.get_energy(xinput=self.x, yinput=self.yp_l, embedding=self.embedding,
                                          reuse=True)

        self.energy_yh = self.energy_yh_ - self.yh_penalty
        self.energy_yl = self.energy_yl_ - self.yl_penalty

        self.yp_ind = self.yp_h_ind
        self.yp = self.yp_h
        self.energy_yp = self.energy_yh


        self.energy_ygradient = tf.gradients(self.energy_yp, self.yp_ind)[0]

        self.diff = (self.value_h - self.value_l) * self.margin_weight_ph
        self.objective = tf.reduce_sum(tf.maximum(-self.energy_yh + self.energy_yl + self.diff, 0.0)) \
                         + self.config.l2_penalty * self.get_l2_loss()

        self.num_update = tf.reduce_sum(tf.cast(
            (self.diff > (self.energy_yh - self.energy_yl)), tf.float32))

        self.vh_sum = tf.reduce_mean(self.value_h)
        self.vl_sum = tf.reduce_mean(self.value_l)
        self.eh_sum = tf.reduce_mean(self.energy_yh)
        self.el_sum = tf.reduce_mean(self.energy_yl)

        grads_vals = self.optimizer.compute_gradients(self.objective)

        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_vals]
        self.grad_norm = tf.constant(0.0)
        self.train_step = self.optimizer.apply_gradients(capped_gvs)
        return self

    def construct_embedding(self, embedding_size, vocabulary_size):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocabulary_size, self.embedding_size])

        with tf.variable_scope(self.config.spen_variable_scope) as scope:
            self.embedding = tf.get_variable("emb", shape=[self.vocabulary_size, self.embedding_size], dtype=tf.float32,
                                             initializer=tfi.zeros(), trainable=True)
        self.embedding_init = self.embedding.assign(self.embedding_placeholder)

        return self

    def construct(self, training_type=TrainingType.Rank_Based):
        if training_type == TrainingType.Rank_Based:
            return self.rank_based_training()
        elif training_type == TrainingType.Value_Matching:
            return self.value_match_training()
        else:
            raise NotImplementedError

    def project_simplex_norm(self, y_ind):
        dim = self.config.dimension
        yd = np.reshape(y_ind, (-1, self.config.output_num, dim))
        eps = np.full(shape=np.shape(yd), fill_value=1e-10)
        y_min = np.min(yd, axis=2)
        y_min_all = np.reshape(np.repeat(y_min, dim), (-1, self.config.output_num, dim))
        yd_pos = yd - y_min_all
        yd_sum = np.reshape(np.repeat(np.sum(yd_pos, 2), dim), (-1, self.config.output_num, dim))
        yd_sum = yd_sum + eps
        yd_norm = np.divide(yd_pos, yd_sum)
        return np.reshape(yd_norm, (-1, self.config.output_num * dim))


    def evaluate(self, xinput=None, yinput=None, yt=None):
        raise NotImplementedError


    def search(self, xinput, ymap, ysoft, yt=None):
        raise NotImplementedError

    def inference(self, xinput=None, yinput=None, yinit=None, inf_iter=None, ascent=True, train=False):
        size = np.shape(xinput)[0]

        yp_ind = np.random.uniform(0, 1, (size, self.config.output_num * self.config.dimension))

        if inf_iter is None:
            inf_iter = self.config.inf_iter
        tflearn.is_training(is_training=train, session=self.sess)
        yp_a = []
        g_m = np.zeros(np.shape(yp_ind))
        alpha = self.config.alpha
        mean = np.zeros(shape=np.shape(yp_ind))
        it = 0
        while it < inf_iter:

            feed_dict = {self.x: xinput, self.yp_ind: yp_ind,
                         self.margin_weight_ph: self.config.margin_weight,
                         self.inf_penalty_weight_ph: self.config.inf_penalty,
                         self.bs_ph: xinput.shape[0],
                         self.is_training: 1.0 if train else 0.0,
                         self.dropout_ph: self.config.dropout}
            yp = self.sess.run(self.yp_h, feed_dict=feed_dict)

            feed_dict = {self.x: xinput, self.yp_ind: yp,
                         self.margin_weight_ph: self.config.margin_weight,
                         self.inf_penalty_weight_ph: self.config.inf_penalty,
                         self.bs_ph: xinput.shape[0],
                         self.is_training: 1.0 if train else 0.0,
                         self.dropout_ph: self.config.dropout}

            g, e = self.sess.run([self.inf_gradient, self.inf_objective], feed_dict=feed_dict)

            #             g = np.clip(g, a_min=-1.0, a_max=1.0)
            gnorm = np.linalg.norm(g, axis=1)
            if train:
                noise = np.random.normal(mean, self.config.noise_rate, size=np.shape(g))
            else:
                noise = np.zeros(shape=np.shape(g))
            g_m = alpha * (g+noise) + (1 - alpha) * g_m
            if ascent:
                yp_ind = yp_ind + (self.config.inf_rate) * (g_m)
            else:
                yp_ind = yp_ind - self.config.inf_rate * (g_m)
            yp_a.append(np.reshape(yp, (-1, self.config.output_num * self.config.dimension)))
            it += 1
        return np.array(yp_a)


    def get_training_points(self, xinput=None, yinput=None, yinit=None, inf_iter=None, ascent=True):
        self.inf_objective = self.energy_yp
        self.inf_gradient = self.energy_ygradient

        y_a = self.inference(xinput=xinput, yinit=yinit, train=True, ascent=ascent, inf_iter=inf_iter)
        y_ans = y_a[-1]
        yp = np.reshape(y_ans, (-1, self.config.output_num, self.config.dimension))

        if self.config.use_search:
            y_better, found = self.search(xinput, np.argmax(yp, 2), yp, yt=yinput)
        else:
            y_better = yinput
            found = np.ones(yp.shape[0])
        yb = self.var_to_indicator(y_better)
        if self.config.loglevel > 30:
            en_better = np.array(self.sess.run(self.inf_objective,
                                               feed_dict={self.x: xinput,
                                                          self.yp_ind: np.reshape(yb, (
                                                              -1, self.config.output_num * self.config.dimension)),
                                                          self.inf_penalty_weight_ph: self.config.inf_penalty,
                                                          self.is_training: 1.0,
                                                          self.dropout_ph: self.config.dropout}))
            en_p = np.array(self.sess.run(self.inf_objective,
                                          feed_dict={self.x: xinput,
                                                     self.yp_ind: np.reshape(y_ans, (
                                                         -1, self.config.output_num * self.config.dimension)),
                                                     self.inf_penalty_weight_ph: self.config.inf_penalty,
                                                     self.is_training: 1.0,

                                                     self.dropout_ph: self.config.dropout}))
        fh = []
        fl = []
        yh = []
        yl = []
        x = []

        for i in range(yp.shape[0]):

            if found[i] > 0:
                if yinput is not None:
                    fp = self.evaluate(xinput=xinput[i], yinput=np.expand_dims(np.argmax(yp[i], 1), 0),
                                       yt=np.expand_dims(yinput[i], 0))
                    fb = self.evaluate(xinput=xinput[i], yinput=np.expand_dims(y_better[i], 0),
                                       yt=np.expand_dims(yinput[i], 0))
                else:
                    fp = self.evaluate(xinput=np.expand_dims(xinput[i], 0),
                                       yinput=np.expand_dims(np.argmax(yp[i], 1), 0))
                    fb = self.evaluate(xinput=np.expand_dims(xinput[i], 0), yinput=np.expand_dims(y_better[i], 0))

                if self.config.loglevel > 30:
                    print(i, fp[0], fb[0], en_p[i], en_better[i])

                fh.append(fb[0])
                fl.append(fp[0])
                yh.append(yb[i])

                yl.append(y_ans[i])
                x.append(xinput[i])

        x = np.array(x)
        fh = np.array(fh)
        fl = np.array(fl)
        yh = np.array(yh)
        yl = np.array(yl)

        return x, yh, yl, fh, fl

    def get_first_large_consecutive_diff(self, xinput=None, yinput=None, yinit=None, inf_iter=None, ascent=True):
        self.inf_objective = self.energy_yp
        self.inf_gradient = self.energy_ygradient

        y_a = self.inference(xinput=xinput, yinit=yinit, train=True, ascent=ascent, inf_iter=inf_iter)


        y_a = y_a[1:]


        en_a = np.array([self.sess.run(self.inf_objective,
                                       feed_dict={self.x: xinput,
                                                  self.yp_ind: np.reshape(y_i, (
                                                      -1, self.config.output_num * self.config.dimension)),
                                                  self.inf_penalty_weight_ph: self.config.inf_penalty,
                                                  self.dropout_ph: self.config.dropout})
                         for y_i in y_a])

        f_a = np.array([self.evaluate(xinput=xinput, yinput=np.argmax(np.reshape(y_i, (-1, self.config.output_num, self.config.dimension)),2), yt=yinput) for y_i in y_a])
        if self.config.loglevel > 25:
            for t in range(xinput.shape[0]):
                print(t, f_a[-2][t], f_a[-1][t], en_a[-2][t], en_a[-1][t])

        size = np.shape(xinput)[0]
        t = np.array(range(size))
        f1 = []
        f2 = []
        y1 = []
        y2 = []
        x = []
        k = 0
        it = np.shape(y_a)[0]
        for k in range(it - 1):
            for i in t:
                # if found[i] < 1:
                #     continue
                if f_a[k, i] > f_a[k + 1, i]:
                    i_h = k
                    i_l = k + 1
                else:
                    i_l = k
                    i_h = k + 1

                f_h = f_a[i_h, i]
                f_l = f_a[i_l, i]
                e_h = en_a[i_h, i]
                e_l = en_a[i_l, i]

                violation = (f_h - f_l) * self.config.margin_weight - e_h + e_l
                if violation > 0 and f_h > f_l:
                # if found[i] > 0:
                    f1.append(f_h)
                    f2.append(f_l)
                    y1.append((y_a[i_h, i, :]))
                    y2.append((y_a[i_l, i, :]))
                    x.append(xinput[i, :])

        x = np.array(x)
        f1 = np.array(f1)
        f2 = np.array(f2)
        y1 = np.array(y1)
        y2 = np.array(y2)

        return x, y1, y2, f1, f2



    def get_all_diff(self, xinput=None, yinput=None, yinit=None, inf_iter=None, ascent=True):
        self.inf_objective = self.energy_yp
        self.inf_gradient = self.energy_ygradient
        y_a = self.inference(xinput=xinput, inf_iter=inf_iter, train=True, ascent=ascent)

        yp_a = np.array([self.sess.run(self.yp_h, feed_dict={self.yp_h_ind: y_i}) for y_i in y_a])

        en_a = np.array([self.sess.run(self.inf_objective,
                                       feed_dict={self.x: xinput,
                                                  self.yp_ind: np.reshape(y_i, (
                                                  -1, self.config.output_num * self.config.dimension)),
                                                  self.inf_penalty_weight_ph: self.config.inf_penalty,
                                                  self.dropout_ph: self.config.dropout})
                         for y_i in yp_a])


        f_a = np.array([self.evaluate(xinput=xinput, yinput=np.argmax(np.reshape(y_i, (-1, self.config.output_num, self.config.dimension)),2), yt=np.argmax(np.reshape(yinput, (-1, self.config.output_num, self.config.dimension)),2)) for y_i in yp_a])

        e_t = self.sess.run(self.inf_objective,
                            feed_dict={self.x: xinput,
                                       self.yp_ind: np.reshape(yinput, (
                                           -1, self.config.output_num * self.config.dimension)),
                                       self.inf_penalty_weight_ph: self.config.inf_penalty,
                                       self.dropout_ph: self.config.dropout})

        size = np.shape(xinput)[0]
        t = np.array(range(size))
        y = []
        yp = []
        x = []
        f1 = []
        f2 = []
        it = np.shape(y_a)[0]
        for k in range(it-1):
            for i in t:

                violation = (f_a[k+1,i]-f_a[k, i]) * self.config.margin_weight - e_t[i] + en_a[k, i]
                # print(e_t[i], en_a[k, i], ce_a[k, i], violation)
                if violation > 0:
                    yp.append((y_a[k, i, :]))
                    x.append(xinput[i, :])
                    y.append(yinput[i, :])
                    f1.append(f_a[k+1, i])
                    f2.append(f_a[k, i])
        x = np.array(x)
        y = np.array(y)
        yp = np.array(yp)
        f1 = np.array(f1)
        f2 = np.array(f2)

        return x, y, yp


    def soft_predict(self, xinput=None, yinit=None, train=False, inf_iter=None, ascent=True, end2end=False):
        tflearn.is_training(is_training=train, session=self.sess)
        self.inf_objective = self.energy_yp
        self.inf_gradient = self.energy_ygradient
        y_a = self.inference(xinput=xinput, yinit=yinit, inf_iter=inf_iter, train=train, ascent=ascent)
        y_ans = y_a[-1]
        yp = np.reshape(y_ans, (-1, self.config.output_num, self.config.dimension))
        return yp



    def map_predict(self, xinput=None, yinit=None, train=False, inf_iter=None, ascent=True, end2end=False, continuous=False):
        yp = self.soft_predict(xinput=xinput, yinit=yinit, train=train, inf_iter=inf_iter, ascent=ascent,
                               end2end=end2end)
        if self.config.dimension == 1:
            return np.squeeze(yp)
        else:
            return np.argmax(yp, 2)



    def train_batch(self, xbatch=None, ybatch=None, verbose=0):
        raise NotImplementedError

    def train_unsupervised_rb_batch(self, xbatch=None, ybatch=None, yinit=None, verbose=0):
        tflearn.is_training(True, self.sess)

        x_b, y_h, y_l, l_h, l_l = self.get_first_large_consecutive_diff(xinput=xbatch, yinput=ybatch, ascent=True)
        dist = np.linalg.norm(np.reshape(y_h, y_l.shape) - y_l)
        total = np.size(l_h)
        if l_l.shape[0] <= 0:
            print ("skip")
            return

        _, o1, g, n1, v1, v2, e1, e2 = self.sess.run(
            [self.train_step, self.objective, self.grad_norm, self.num_update, self.vh_sum, self.vl_sum,
             self.eh_sum, self.el_sum],
            feed_dict={self.x: x_b,
                       self.yp_h_ind: np.reshape(y_h, (-1, self.config.output_num * self.config.dimension)),
                       self.yp_l_ind: np.reshape(y_l, (-1, self.config.output_num * self.config.dimension)),
                       self.value_l: l_l,
                       self.value_h: l_h,
                       self.learning_rate_ph: self.config.learning_rate,
                       self.dropout_ph: self.config.dropout,
                       self.inf_penalty_weight_ph: self.config.inf_penalty,
                       self.is_training: 1.0,
                       self.bs_ph: x_b.shape[0],
                       self.margin_weight_ph: self.config.margin_weight})

        if verbose > 0:
            print(self.train_iter, o1, g, v1, v2, e1, e2, dist, np.shape(xbatch)[0], np.shape(x_b)[0],
                  np.average(l_l))
        return np.shape(x_b)[0]


    def train_unsupervised_sg_batch(self, xbatch=None, ybatch=None, yinit=None, verbose=0):
        tflearn.is_training(True, self.sess)

        x_b, y_h, y_l, l_h, l_l = self.get_training_points(xinput=xbatch, yinput=ybatch, yinit=yinit,ascent=True)
        dist = np.linalg.norm(np.reshape(y_h, y_l.shape) - y_l)
        total = np.size(l_h)
        if l_l.shape[0] <= 0:
            print "skip"
            return
        _, o1, g, n1, v1, v2, e1, e2 = self.sess.run(
            [self.train_step, self.objective, self.grad_norm, self.num_update, self.vh_sum, self.vl_sum,
             self.eh_sum, self.el_sum],
            feed_dict={self.x: x_b,
                       self.yp_h_ind: np.reshape(y_h, (-1, self.config.output_num * self.config.dimension)),
                       self.yp_l_ind: np.reshape(y_l, (-1, self.config.output_num * self.config.dimension)),
                       self.value_l: l_l,
                       self.value_h: l_h,
                       self.learning_rate_ph: self.config.learning_rate,
                       self.dropout_ph: self.config.dropout,
                       self.inf_penalty_weight_ph: self.config.inf_penalty,
                       self.is_training : 1.0,
                       self.bs_ph : x_b.shape[0],
                       self.margin_weight_ph: self.config.margin_weight})

        self.log_const[self.train_iter] = x_b.shape[0]

        if verbose > 0:
            print(self.train_iter, o1, g, v1, v2, e1, e2, dist, np.shape(xbatch)[0], np.shape(x_b)[0], np.average(l_l))
        return






    def save(self, path):
        self.saver.save(self.sess, path)

    def restore(self, path):
        self.saver.restore(self.sess, path)

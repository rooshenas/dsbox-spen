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
        self.embedding = None

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

        for v in self.pred_variables():
            print(v)

    def spen_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen")

    def energy_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen/en")

    def energy_g_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen/en/en.g")

    def fnet_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen/fx")

    def pred_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="pred")

    def predx_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="pred/xpred")

    def predh_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="pred/hpred")

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

    def get_loss(self, yt, yp):
        raise NotImplementedError

    def f1_loss(self, yt, yp):
        l = -tf.reduce_sum((tf.reduce_sum(yt * tf.log(tf.maximum(yp, 1e-20)), 1)
                            + 0.1 * tf.reduce_sum((1. - yt) * tf.log(tf.maximum(1. - yp, 1e-20)), 1)))
        yp = tf.reshape(yp, [-1, self.config.output_num, self.config.dimension])
        yt = tf.reshape(yt, [-1, self.config.output_num, self.config.dimension])
        yp_ones = yp[:, :, 1]
        yt_ones = yt[:, :, 1]
        intersect = tf.reduce_sum(tf.minimum(yt_ones, yp_ones), 1)
        return -tf.reduce_sum(
            2 * intersect / (tf.reduce_sum(yt_ones, 1) + tf.reduce_sum(yp_ones, 1))) + l - 1.2 * tf.reduce_sum(yp_ones)

    def ce_loss(self, yt, yp):
        eps = 1e-30
        ypc = tf.reshape(yp, (-1, self.config.output_num, self.config.dimension))
        # yp = tf.clip_by_value(yp, clip_value_min=eps, clip_value_max=1.0 - eps )
        ytc = tf.reshape(yt, (-1, self.config.output_num, self.config.dimension))
        # l =  tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=yt, logits=yp))
        l = -tf.reduce_sum(yt * tf.log(tf.maximum(yp, eps))) \
            - tf.reduce_sum((1. - yt) * tf.log(tf.maximum(1. - yp, eps)))
        if self.config.dimension == 1:
            self.syp = tf.reduce_mean(yp)
            self.syp_t = tf.reduce_mean(yp)
            self.pred = yp
            self.gt = yt
        else:
            self.syp = tf.reduce_mean(tf.reduce_max(ypc, -1))
            self.syp_t = tf.reduce_mean(tf.reduce_max(tf.multiply(ypc, ytc), -1))

            self.pred = tf.argmax(ypc, axis=2)
            self.gt = tf.argmax(ytc, axis=2)

        return l

    def ce_sym_loss(self, yt, yp):
        l = -tf.reduce_sum((tf.reduce_sum(yt * tf.log(tf.maximum(yp, 1e-20)), 1) \
                            + tf.reduce_sum((1. - yt) * tf.log(tf.maximum(1. - yp, 1e-20)), 1)))
        if self.config.dimension == 1:
            self.syp = tf.reduce_mean(yp)
            self.syp_t = tf.reduce_mean(yp)
            self.pred = yp
            self.gt = yt

        return l

    def ce_en_loss(self, yt, yp):
        eps = 1e-30
        ypc = tf.reshape(yp, (-1, self.config.output_num, self.config.dimension))
        # yp = tf.clip_by_value(yp, clip_value_min=eps, clip_value_max=1.0 - eps )
        ytc = tf.reshape(yt, (-1, self.config.output_num, self.config.dimension))
        # l =  tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=yt, logits=yp))
        l = -tf.reduce_sum(ytc * tf.log(tf.maximum(ypc, eps))) - 0.1 * tf.reduce_sum(ypc * tf.log(tf.maximum(ypc, eps)))
        #            -tf.reduce_sum((1. - ytc) * tf.log(tf.maximum(1. - ypc , eps)))
        self.syp = tf.reduce_mean(tf.reduce_max(ypc, -1))
        self.syp_t = tf.reduce_mean(tf.reduce_max(tf.multiply(ypc, ytc), -1))

        self.pred = tf.argmax(ypc, axis=2)
        self.gt = tf.argmax(ytc, axis=2)

        return l

    def mse_loss(self, yt, yp):
        # yt_max = tf.cast(tf.argmax(tf.reshape(yt, (-1, self.config.output_num, self.config.dimension)), axis=2), tf.float32)
        # yp_max = tf.cast(tf.argmax(tf.reshape(yp, (-1, self.config.output_num, self.config.dimension)), axis=2), tf.float32)
        self.syp = tf.reduce_mean(yp)
        self.syp_t = tf.reduce_mean(yt)
        self.pred = yp
        self.gt = yt
        eps = 1e-20

        l = tf.reduce_mean(tf.square((yt - yp) * 255.0))
        return l

    def biased_loss(self, yt, yp):
        l = -tf.reduce_sum((tf.reduce_sum(yt * tf.log(tf.maximum(yp, 1e-20)), 1) \
                            + tf.reduce_sum((1. - yt) * tf.log(tf.maximum(1. - yp, 1e-20)), 1)))
        yp = tf.reshape(yp, [-1, self.config.output_num, self.config.dimension])
        yp_zeros = yp[:, :, 0]
        yp_ones = yp[:, :, 1]
        en = tf.reduce_sum(yp_ones * tf.log(yp_ones), 1)
        return l + 1.2 * (tf.reduce_sum(yp_zeros) - tf.reduce_sum(yp_ones)) + 0.0 * tf.reduce_sum(en)

    def get_initialization_net(self, xinput, output_size, reuse=False):
        raise NotImplementedError

    def cll_training(self):
        self.margin_weight_ph = tf.placeholder(tf.float32, shape=[], name="Margin")
        self.inf_penalty_weight_ph = tf.placeholder(tf.float32, shape=[], name="InfPenalty")
        self.yp_h_ind = tf.placeholder(tf.float32,
                                       shape=[None, self.config.output_num * self.config.dimension],
                                       name="YP_H")


        yp_ind_sm_h = tf.nn.softmax(tf.reshape(self.yp_h_ind, [-1, self.config.output_num, self.config.dimension]))
        self.yp_h = tf.reshape(yp_ind_sm_h, [-1, self.config.output_num * self.config.dimension])
        self.energy_yh = self.get_energy(xinput=self.x, yinput=self.yp_h, embedding=self.embedding, reuse=False)
        self.objective = -tf.reduce_sum(self.energy_yh)
        self.yp_ind = self.yp_h_ind
        self.yp = self.yp_h
        self.energy_yp = self.energy_yh

        self.energy_ygradient = tf.gradients(self.energy_yp, self.yp_ind)[0]
        self.train_step = self.optimizer.minimize(self.objective)


    def value_match_training(self):
        self.margin_weight_ph = tf.placeholder(tf.float32, shape=[], name="Margin")
        self.inf_penalty_weight_ph = tf.placeholder(tf.float32, shape=[], name="InfPenalty")
        self.yp_ind = tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension],
                                     name="Output")
        self.v_ind = tf.placeholder(tf.float32, shape=[None],
                                    name="value")

        self.y_penalty = self.inf_penalty_weight_ph * tf.reduce_sum(tf.square(self.yp_ind), 1)

        self.energy_yp = tf.nn.sigmoid(
            self.get_energy(xinput=self.x, yinput=self.yp_ind, embedding=self.embedding))  # - self.y_penalty

        self.energy_ygradient = tf.gradients(self.energy_yp, self.yp_ind)[0]
        self.newce = -tf.reduce_sum(self.v_ind * tf.log(tf.maximum(self.energy_yp, 1e-20))) \
                     - tf.reduce_sum(1 - self.v_ind * tf.log(tf.maximum(1 - self.energy_yp, 1e-20)))

        # self.error = tf.reduce_sum(tf.square(self.energy_yp - self.v_ind))
        self.en = tf.reduce_sum(self.energy_yp)
        self.error = self.newce
        self.objective = self.error \
                         + self.config.l2_penalty * self.get_l2_loss()

        self.train_step = self.optimizer.minimize(self.objective)

    def end2end_training(self):
        self.inf_penalty_weight_ph = tf.placeholder(tf.float32, shape=[], name="InfPenalty")
        self.h = tf.placeholder(tf.float32, shape=[None, self.config.hidden_num], name="hinput")
        try:
            h_start = self.get_initialization_net(self.x, self.config.hidden_num)
        except:
            self.h = tf.placeholder(tf.float32, shape=[None, self.config.hidden_num], name="hinput")
            h_start = self.h
        # except:
        #  raise  NotImplementedError("Should have used init model")

        # self.yp_h  = self.get_prediction_net(input=tf.concat((h_start, h_start), axis=1), xinput=self.x)


        self.yt_ind = tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension],
                                     name="OutputYT")
        # self.h = self.get_feature_net(self.x, self.config.hidden_num, embedding=self.embedding)

        self.h_penalty = self.inf_penalty_weight_ph * tf.reduce_sum(tf.square(h_start), 1)
        self.avg_h = tf.reduce_mean(tf.square(h_start))
        # self.energy_h = self.get_energy(xinput=self.x, yinput=self.h, embedding=self.embedding) - self.h_penalty
        h_current = h_start
        self.objective = 0.0
        self.h_ar = [h_start]

        self.yp_ar = []
        self.en_ar = []
        self.g_ar = []
        self.pen_ar = []
        # self.objective = self.get_loss(self.yt_ind, self.yp_h)
        for i in range(int(self.config.inf_iter)):
            # penalty_current = self.inf_penalty_weight_ph* tf.reduce_sum(tf.square(h_current),1)
            energy_current = self.get_energy(xinput=self.x, yinput=h_current, embedding=self.embedding,
                                             reuse=False if i == 0 else True)  # - penalty_current
            g = tf.gradients(energy_current, h_current)[0]
            self.en_ar.append(energy_current)
            self.g_ar.append(g)
            # self.pen_ar.append(penalty_current)
            noise = tf.random_normal(shape=tf.shape(g),
                                     stddev=self.config.noise_rate * tf.norm(g) / tf.sqrt(tf.cast(i, tf.float32) + 1.0))

            h_next = h_current + self.config.inf_rate * (
            self.config.inf_rate / tf.sqrt(tf.cast(i, tf.float32) + 1.0)) * tf.cond(self.is_training > 0.0,
                                                                                    lambda: g + noise, lambda: g)
            h_current = h_next
            h_extend = tf.concat((h_current, h_start), axis=1)
            yp_current = self.get_prediction_net(input=h_extend, xinput=self.x, reuse=False if i == 0 else True)
            ind = tf.reshape(yp_current, [-1, self.config.output_num * self.config.dimension])
            l = self.get_loss(self.yt_ind, ind)
            self.h_ar.append(h_current)
            self.objective = (1.0 - self.config.alpha) * self.objective + self.config.alpha * l
            self.yp_ar.append(yp_current)

        # self.opjective = l
        self.objective += self.config.l2_penalty * self.get_l2_loss()
        if self.config.dimension > 1:
            self.yp = tf.reshape(self.yp_ar[-1], [-1, self.config.output_num, self.config.dimension])
        else:
            self.yp = self.yp_ar[-1]
        # self.get_prediction_net(input=self.h_state)

        # self.yp_ind = tf.reshape(self.yp, [-1, self.config.output_num * self.config.dimension], name="reshaped")
        # self.objective = -tf.reduce_sum(self.yt_ind * tf.log( tf.maximum(self.yp_ind, 1e-20)))
        self.train_all_step = self.optimizer.minimize(self.objective)
        self.train_step = self.optimizer.minimize(self.objective, var_list=self.spen_variables())

        # self.yp_mlp = self.get_prediction_net(input=self.h0, xinput=self.x, reuse=True)
        # self.objective_mlp = self.get_loss(self.yt_ind, self.yp_mlp) + self.config.l2_penalty * self.get_l2_loss()
        self.train_pred_step = self.optimizer.minimize(self.objective, var_list=self.pred_variables())
        # predxh = self.predh_variables() + self.predx_variables()
        # hspen = self.predh_variables() + self.spen_variables()
        # self.train_hspen_step = self.optimizer.minimize(self.objective, var_list=hspen)
        # self.train_predhx_step = self.optimizer.minimize(self.objective, var_list=predxh)

    def ssvm_training(self):
        self.margin_weight_ph = tf.placeholder(tf.float32, shape=[], name="Margin")
        self.inf_penalty_weight_ph = tf.placeholder(tf.float32, shape=[], name="InfPenalty")
        self.yp_ind = tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension],
                                     name="OutputYP")
        self.yt_ind = tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension],
                                     name="OutputYT")

        self.y_penalty = self.inf_penalty_weight_ph * tf.reduce_sum(tf.square(self.yp_ind), 1)
        self.yt_penalty = self.inf_penalty_weight_ph * tf.reduce_sum(tf.square(self.yt_ind), 1)

        self.energy_yp = self.get_energy(xinput=self.x, yinput=self.yp_ind, embedding=self.embedding) - self.y_penalty
        self.energy_yt = self.get_energy(xinput=self.x, yinput=self.yt_ind, embedding=self.embedding,
                                         reuse=True) - self.yt_penalty

        yp_ind_2 = tf.reshape(self.yp_ind, [-1, self.config.output_num, self.config.dimension], name="res1")
        yp_ind_sm = tf.nn.softmax(yp_ind_2, name="sm")
        self.yp = tf.reshape(yp_ind_sm, [-1, self.config.output_num * self.config.dimension], name="res2")

        self.ce = -tf.reduce_sum(self.yt_ind * tf.log(tf.maximum(self.yp, 1e-20)), 1)
        self.en = -tf.reduce_sum(self.yp * tf.log(tf.maximum(self.yp, 1e-20)), 1)

        self.loss_augmented_energy = self.energy_yp + self.ce * self.margin_weight_ph  # + self.y_penalty
        self.loss_augmented_energy_ygradient = tf.gradients(self.loss_augmented_energy, self.yp_ind)[0]

        self.energy_ygradient = tf.gradients(self.energy_yp, self.yp_ind)[0]

        self.objective = tf.reduce_sum(
            tf.maximum(self.ce * self.config.margin_weight + self.energy_yp - self.energy_yt, 0.0)) \
                         + self.config.l2_penalty * self.get_l2_loss()

        self.num_update = tf.reduce_sum(
            tf.cast(self.ce * self.margin_weight_ph > self.energy_yt - self.energy_yp, tf.float32))
        self.total_energy_yt = tf.reduce_sum(self.energy_yt)
        self.total_energy_yp = tf.reduce_sum(self.energy_yp)

        self.train_step = self.optimizer.minimize(self.objective, var_list=self.spen_variables())

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

        # self.yh_penalty =  self.inf_penalty_weight_ph * tf.reduce_logsumexp(self.yp_h_ind ,1)
        # self.yl_penalty =  self.inf_penalty_weight_ph * tf.reduce_logsumexp(self.yp_l_ind, 1)

        self.yh_penalty = self.inf_penalty_weight_ph * tf.reduce_sum(tf.square(self.yp_h_ind), 1)
        self.yl_penalty = self.inf_penalty_weight_ph * tf.reduce_sum(tf.square(self.yp_l_ind), 1)

        self.energy_yh_ = self.get_energy(xinput=self.x, yinput=self.yp_h, embedding=self.embedding,
                                          reuse=False)  # - self.yh_penalty
        self.energy_yl_ = self.get_energy(xinput=self.x, yinput=self.yp_l, embedding=self.embedding,
                                          reuse=True)  # - self.yl_penalty

        self.energy_yh = self.energy_yh_ - self.yh_penalty
        self.energy_yl = self.energy_yl_ - self.yl_penalty

        self.yp_ind = self.yp_h_ind
        self.yp = self.yp_h
        self.energy_yp = self.energy_yh

        # self.en = -tf.reduce_sum(self.yp * tf.log( tf.maximum(self.yp, 1e-20)), 1)

        self.energy_ygradient = tf.gradients(self.energy_yp, self.yp_ind)[0]

        # self.objective = tf.reduce_mean( tf.maximum(
        #          (self.value_h - self.value_l)*self.margin_weight_ph - self.energy_yh + self.energy_yl, 0.0)) \
        #
        #                  + self.config.l2_penalty * self.get_l2_loss()
        # ratio = tf.divide(self.value_h-self.value_l, tf.abs(self.value_h))
        self.ce = -tf.reduce_sum(self.yp_h * tf.log(tf.maximum(self.yp_l, 1e-20)), 1)
        self.diff = (self.value_h - self.value_l) * self.margin_weight_ph
        self.objective = tf.reduce_sum(tf.maximum(-self.energy_yh + self.energy_yl + self.diff, 0.0)) \
                         + self.config.l2_penalty * self.get_l2_loss()

        self.num_update = tf.reduce_sum(tf.cast(
            (self.diff > (self.energy_yh - self.energy_yl)), tf.float32))

        self.vh_sum = tf.reduce_mean(self.value_h)
        self.vl_sum = tf.reduce_mean(self.value_l)
        self.eh_sum = tf.reduce_mean(self.energy_yh)
        self.el_sum = tf.reduce_mean(self.energy_yl)
        # self.train_step = self.optimizer.minimize(self.objective, var_list=self.energy_variables())
        # self.train_step = self.optimizer.minimize(self.objective)
        grads_vals = self.optimizer.compute_gradients(self.objective)

        grads = grads_vals[0]

        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_vals]
        #train_op = optimizer.apply_gradients(capped_gvs)
        # self.grad_norm = tf.reduce_mean(tf.norm(grads, 1))
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

    def construct(self, training_type=TrainingType.SSVM):
        # tf.reset_default_graph()
        if training_type == TrainingType.SSVM:
            return self.ssvm_training()
        elif training_type == TrainingType.Rank_Based:
            return self.rank_based_training()
        elif training_type == TrainingType.End2End:
            return self.end2end_training()
        elif training_type == TrainingType.Value_Matching:
            return self.value_match_training()
        elif training_type == TrainingType.CLL:
            return self.cll_training()
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

    def project_indicators(self, y_ind):
        yd = self.indicator_to_var(y_ind)
        yd_norm = self.project_simplex_norm(yd)
        return self.var_to_indicator(yd_norm)

    def gather_numpy(self, y, dim, index):
        """
        https://stackoverflow.com/questions/46065873/how-to-do-scatter-and-gather-operations-in-numpy
        Gathers values along an axis specified by dim.
        For a 3-D tensor the output is specified by:
            out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
            out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
            out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

        :param dim: The axis along which to index
        :param index: A tensor of indices of elements to gather
        :return: tensor of gathered values
        """
        idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
        self_xsection_shape = y.shape[:dim] + y.shape[dim + 1:]
        if idx_xsection_shape != self_xsection_shape:
            raise ValueError("Except for dimension " + str(dim) +
                             ", all dimensions of index and self should be the same size")
        if index.dtype != np.dtype('int_'):
            raise TypeError("The values of index must be integers")
        data_swaped = np.swapaxes(y, 0, dim)
        index_swaped = np.swapaxes(index, 0, dim)
        gathered = np.choose(index_swaped, data_swaped)
        return np.swapaxes(gathered, 0, dim)

    def project_simplex_opt(self, y):
        y = np.reshape(y, (-1, self.config.output_num, self.config.dimension))
        u = np.flip(np.sort(y, axis=-1), axis=-1)
        positions = np.arange(0, u.shape[-1])
        u_sum = (1 - u.cumsum(axis=-1)) / (positions + 1.0)
        ru = np.argmax((((u_sum + u > 0) + 0) * (positions + 1)), axis=-1)
        ru_ind = np.reshape(np.repeat(ru, self.config.dimension, axis=-1), y.shape)
        lambda_ = self.gather_numpy(u_sum, 2, ru_ind)
        yd_norm = np.clip(y + lambda_, a_min=0.0, a_max=1e1000)
        return np.reshape(yd_norm, (-1, self.config.output_num * self.config.dimension))

    def softmax2(self, y, theta=1.0, axis=None):
        y = self.project_simplex_norm(np.reshape(y, (-1, self.config.output_num * self.config.dimension)))
        return np.reshape(y, (-1, self.config.output_num, self.config.dimension))

    def softmax(self, y, theta=1.0, axis=None):

        """
        Compute the softmax of each element along an axis of X.

        Parameters
        ----------
        X: ND-Array. Probably should be floats.
        theta (optional): float parameter, used as a multiplier
            prior to exponentiation. Default = 1.0
        axis (optional): axis to compute values along. Default is the
            first non-singleton axis.

        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
        """

        y = np.reshape(y, (-1, self.config.output_num, self.config.dimension))
        if axis is None:
            axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

        # multiply y against the theta parameter,
        y = y * float(theta)

        # subtract the max for numerical stability
        y = y - np.expand_dims(np.max(y, axis=axis), axis)

        # exponentiate y
        y = np.exp(y)

        # take the sum along the specified axis
        ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

        # finally: divide elementwise
        p = y / ax_sum

        return p

    def get_gradient(self, xinput=None, yinput=None, train=False):
        self.inf_objective = self.energy_yp
        self.inf_gradient = self.energy_ygradient
        tflearn.is_training(is_training=train, session=self.sess)
        yp_ind = np.reshape(yinput, (-1, self.config.output_num * self.config.dimension))

        feed_dict = {self.x: xinput, self.yp_ind: yp_ind,
                     self.margin_weight_ph: self.config.margin_weight,
                     self.inf_penalty_weight_ph: self.config.inf_penalty,
                     self.dropout_ph: self.config.dropout}

        g, e = self.sess.run([self.inf_gradient, self.inf_objective], feed_dict=feed_dict)
        return g, e

    def inference_o(self, xinput=None, yinput=None, yinit=None, inf_iter=None, ascent=True, train=False):
        if self.config.loglevel > 5:
            print("inferece")
        if inf_iter is None:
            inf_iter = self.config.inf_iter
        tflearn.is_training(is_training=train, session=self.sess)
        size = np.shape(xinput)[0]
        if yinput is not None:
            yt_ind = np.reshape(yinput, (-1, self.config.output_num * self.config.dimension))
        if yinit is not None:
            if yinit.shape[0] > 1:
                pass
            else:
                yinit = np.tile(np.reshape(yinit, (1, -1)), xinput.shape[0])
            yp_ind = np.reshape(yinit, (xinput.shape[0], self.config.output_num * self.config.dimension))
        else:
            yp_ind = np.random.uniform(0, 1, (size, self.config.output_num * self.config.dimension))

        # y = np.full((size, self.config.output_num), fill_value=5)
        # y = np.random.randint(0, self.config.dimension, (size, self.config.output_num))
        # y = np.zeros(shape=(size, self.config.output_num))
        # yp_ind = np.reshape(self.var_to_indicator(y), (-1, self.config.output_num* self.config.dimension))
        # yp_ind = np.zeros((size, self.config.output_num * self.config.dimension))
        # yp_ind = self.project_simplex_norm(yp_ind)
        i = 0
        yp_a = []
        g_m = np.zeros(np.shape(yp_ind))
        alpha = 1.0
        mean = np.zeros(shape=np.shape(yp_ind))
        avg_gnorm = 100.0

        # vars = self.sess.run(self.energy_variables())
        # print "W: ", np.sum(np.square(vars))
        it = 0
        while avg_gnorm > self.config.gradient_thresh and it < self.config.inf_iter:
            if yinput is not None:
                feed_dict = {self.x: xinput, self.yp_ind: yp_ind, self.yt_ind: yt_ind,
                             self.margin_weight_ph: self.config.margin_weight,
                             self.inf_penalty_weight_ph: self.config.inf_penalty,
                             self.dropout_ph: self.config.dropout}
            else:
                feed_dict = {self.x: xinput, self.yp_ind: yp_ind,
                             self.margin_weight_ph: self.config.margin_weight,
                             self.inf_penalty_weight_ph: self.config.inf_penalty,
                             self.dropout_ph: self.config.dropout}

            g, e = self.sess.run([self.inf_gradient, self.inf_objective], feed_dict=feed_dict)
            g = np.clip(g, a_min=-1.0, a_max=1.0)
            gnorm = np.linalg.norm(g, axis=1)
            # yp = self.softmax2(np.reshape(yp_ind, (-1, self.config.output_num, self.config.dimension)), axis=2, theta=self.config.temperature)
            avg_gnorm = np.average(gnorm)
            # print avg_gnorm
            # g = np.clip(g,-10, 10)
            if train:
                #noise = np.random.normal(mean, inf_iter*np.abs(g) / math.sqrt((1+i)), size=np.shape(g))
                noise = np.random.normal(mean, self.config.noise_rate*np.average(gnorm) / math.sqrt((1+it)), size=np.shape(g))
            #   #noise = np.random.normal(mean, np.abs(g), size=np.shape(g))
            else:
               noise = np.zeros(shape=np.shape(g))
            #   #noise = np.random.normal(mean, 100.0 / math.sqrt(1+i), size=np.shape(g))
            g_m = alpha * (g+noise) + (1 - alpha) * g_m
            if ascent:
                yp_ind = yp_ind + (self.config.inf_rate) * (g_m)
                # yp_ind = yp_ind + (self.config.inf_rate / math.sqrt((1+it))) * (g+noise)
            else:
                yp_ind = yp_ind - self.config.inf_rate * (g_m)

            # yp = self.softmax(np.reshape(yp_ind, (-1, self.config.output_num, self.config.dimension)), axis=2, theta=1)
            # yp = self.softmax(np.reshape(yp_ind, (-1, self.config.output_num, self.config.dimension)), axis=2, theta=1)
            # yp_ind = self.project_simplex_norm(yp_ind)
            # yp_ind = self.softmax(np.reshape(yp_ind, (-1, self.config.output_num, self.config.dimension)), axis=2, theta=1)
            # yp_a.append(yp_ind)
            # yp_ind = np.reshape(yp_ind, (-1, self.config.output_num*self.config.dimension))
            # yp_proj = self.project_simplex_opt(yp_ind)



            # yp_ind = yp_proj
            # yp_proj = np.reshape(yp_proj, (-1, self.config.output_num, self.config.dimension))
            # yp_a.append(yp_proj)




            if self.config.loglevel > 5:
                # yr = np.reshape(yp_ind, (-1, self.config.output_num, self.config.dimension))
                yp = self.sess.run(self.yp_h, feed_dict={self.yp_h_ind: yp_ind})
                yp = np.reshape(yp, (-1, self.config.output_num, self.config.dimension))
                print(("energy:", np.average(e), "yind:", np.average(np.sum(np.square(yp_ind), 1)),
                      "gnorm:", np.average(gnorm), "yp:", np.average(np.max(yp, 2))))

            # yp_a.append(np.reshape(yp, (-1, self.config.output_num* self.config.dimension)))
            yp_a.append(yp_ind)
            it += 1

        return np.array(yp_a)

    def inference(self, xinput=None, yinput=None, yinit=None, inf_iter=None, ascent=True, train=False):
        if self.config.loglevel > 5:
            print("inferece")
        if inf_iter is None:
            inf_iter = self.config.inf_iter
        tflearn.is_training(is_training=train, session=self.sess)
        size = np.shape(xinput)[0]
        if yinput is not None:
            yt_ind = np.reshape(yinput, (-1, self.config.output_num * self.config.dimension))
        if yinit is not None:
            if yinit.shape[0] > 1:
                pass
            else:
                yinit = np.tile(np.reshape(yinit, (1, -1)), xinput.shape[0])
            yp_ind = np.reshape(yinit, (xinput.shape[0], self.config.output_num * self.config.dimension))
        else:
            yp_ind = np.random.uniform(0, 1, (size, self.config.output_num * self.config.dimension))

        # y = np.full((size, self.config.output_num), fill_value=5)
        # y = np.random.randint(0, self.config.dimension, (size, self.config.output_num))
        # y = np.zeros(shape=(size, self.config.output_num))
        # yp_ind = np.reshape(self.var_to_indicator(y), (-1, self.config.output_num* self.config.dimension))
        # yp_ind = np.zeros((size, self.config.output_num * self.config.dimension))
        # yp_ind = self.project_simplex_norm(yp_ind)
        i = 0
        yp_a = []
        g_m = np.zeros(np.shape(yp_ind))
        alpha = self.config.alpha
        mean = np.zeros(shape=np.shape(yp_ind))
        avg_gnorm = 100.0

        # vars = self.sess.run(self.energy_variables())
        # print "W: ", np.sum(np.square(vars))
        it = 0
        while avg_gnorm > self.config.gradient_thresh and it < self.config.inf_iter:

            feed_dict = {self.x: xinput, self.yp_ind: yp_ind,
                         self.margin_weight_ph: self.config.margin_weight,
                         self.inf_penalty_weight_ph: self.config.inf_penalty,
                         self.dropout_ph: self.config.dropout}
            yp = self.sess.run(self.yp_h, feed_dict=feed_dict)

            if yinput is not None:
                feed_dict = {self.x: xinput, self.yp_ind: yp_ind, self.yt_ind: yt_ind,
                             self.margin_weight_ph: self.config.margin_weight,
                             self.inf_penalty_weight_ph: self.config.inf_penalty,
                             self.dropout_ph: self.config.dropout}
            else:
                feed_dict = {self.x: xinput, self.yp_ind: yp,
                             self.margin_weight_ph: self.config.margin_weight,
                             self.inf_penalty_weight_ph: self.config.inf_penalty,
                             self.dropout_ph: self.config.dropout}

            g, e = self.sess.run([self.inf_gradient, self.inf_objective], feed_dict=feed_dict)
            # print yp.shape

            g = np.clip(g, a_min=-1000.0, a_max=1000.0)
            gnorm = np.linalg.norm(g, axis=1)
            # yp = self.softmax2(np.reshape(yp_ind, (-1, self.config.output_num, self.config.dimension)), axis=2, theta=self.config.temperature)
            avg_gnorm = np.average(gnorm)
            # print avg_gnorm
            # g = np.clip(g,-10, 10)
            if train:
                # noise = np.random.normal(mean, inf_iter*np.abs(g) / math.sqrt((1+i)), size=np.shape(g))
                noise = np.random.normal(mean, self.config.noise_rate * np.average(gnorm) / math.sqrt((1 + it)),
                                         size=np.shape(g))
            # #noise = np.random.normal(mean, np.abs(g), size=np.shape(g))
            else:
                noise = np.zeros(shape=np.shape(g))
            # #noise = np.random.normal(mean, 100.0 / math.sqrt(1+i), size=np.shape(g))
            g_m = alpha * (g + noise) + (1 - alpha) * g_m
            if ascent:
                yp_ind = yp_ind + (self.config.inf_rate) * (g_m)
                # yp_ind = yp_ind + (self.config.inf_rate / math.sqrt((1+it))) * (g+noise)
            else:
                yp_ind = yp_ind - self.config.inf_rate * (g_m)

            if self.config.loglevel > 5:
                # yr = np.reshape(yp_ind, (-1, self.config.output_num, self.config.dimension))
                ypn = self.softmax(yp_ind, axis=-1)
                print(("energy:", np.average(e), "yind:", np.average(np.sum(np.square(yp_ind), 1)),
                      "gnorm:", np.average(gnorm), "yp:", np.average(np.max(ypn, 2))))

            # yp_a.append(np.reshape(yp, (-1, self.config.output_num* self.config.dimension)))
            yp_a.append(np.reshape(yp, (-1, self.config.output_num * self.config.dimension)))
            it += 1

        return np.array(yp_a)




    def inference_old(self, xinput=None, yinput=None, yinit=None, inf_iter=None, ascent=True, train=False):
        if self.config.loglevel > 5:
            print("inferece")
        if inf_iter is None:
            inf_iter = self.config.inf_iter
        tflearn.is_training(is_training=train, session=self.sess)
        size = np.shape(xinput)[0]
        if yinput is not None:
            yt_ind = np.reshape(yinput, (-1, self.config.output_num * self.config.dimension))
        if yinit is not None:
            if yinit.shape[0] > 1:
                pass
            else:
                yinit = np.tile(np.reshape(yinit, (1, -1)), xinput.shape[0])
            yp_ind = np.reshape(yinit, (xinput.shape[0], self.config.output_num * self.config.dimension))
        else:
            yp_ind = np.random.uniform(0, 1, (size, self.config.output_num * self.config.dimension))

        # y = np.full((size, self.config.output_num), fill_value=5)
        # y = np.random.randint(0, self.config.dimension, (size, self.config.output_num))
        # y = np.zeros(shape=(size, self.config.output_num))
        # yp_ind = np.reshape(self.var_to_indicator(y), (-1, self.config.output_num* self.config.dimension))
        # yp_ind = np.zeros((size, self.config.output_num * self.config.dimension))
        # yp_ind = self.project_simplex_norm(yp_ind)
        i = 0
        yp_a = []
        g_m = np.zeros(np.shape(yp_ind))
        alpha = self.config.alpha
        mean = np.zeros(shape=np.shape(yp_ind))
        avg_gnorm = 100.0

        # vars = self.sess.run(self.energy_variables())
        # print "W: ", np.sum(np.square(vars))
        it = 0
        while avg_gnorm > self.config.gradient_thresh and it < self.config.inf_iter:

            feed_dict = {self.x: xinput, self.yp_ind: yp_ind,
                         self.margin_weight_ph: self.config.margin_weight,
                         self.inf_penalty_weight_ph: self.config.inf_penalty,
                         self.dropout_ph: self.config.dropout}
            yp = self.sess.run(self.yp_h, feed_dict=feed_dict)

            if yinput is not None:
                feed_dict = {self.x: xinput, self.yp_ind: yp_ind, self.yt_ind: yt_ind,
                             self.margin_weight_ph: self.config.margin_weight,
                             self.inf_penalty_weight_ph: self.config.inf_penalty,
                             self.dropout_ph: self.config.dropout}
            else:
                feed_dict = {self.x: xinput, self.yp_ind: yp,
                             self.margin_weight_ph: self.config.margin_weight,
                             self.inf_penalty_weight_ph: self.config.inf_penalty,
                             self.dropout_ph: self.config.dropout}

            g, e = self.sess.run([self.inf_gradient, self.inf_objective], feed_dict=feed_dict)
            # print yp.shape

            g = np.clip(g, a_min=-1.0, a_max=1.0)
            gnorm = np.linalg.norm(g, axis=1)
            # yp = self.softmax2(np.reshape(yp_ind, (-1, self.config.output_num, self.config.dimension)), axis=2, theta=self.config.temperature)
            avg_gnorm = np.average(gnorm)
            # print avg_gnorm
            # g = np.clip(g,-10, 10)
            if train:
                # noise = np.random.normal(mean, inf_iter*np.abs(g) / math.sqrt((1+i)), size=np.shape(g))
                noise = np.random.normal(mean, self.config.noise_rate * np.average(gnorm) / math.sqrt((1 + it)),
                                         size=np.shape(g))
            # #noise = np.random.normal(mean, np.abs(g), size=np.shape(g))
            else:
                noise = np.zeros(shape=np.shape(g))
            # #noise = np.random.normal(mean, 100.0 / math.sqrt(1+i), size=np.shape(g))
            g_m = alpha * (g + noise) + (1 - alpha) * g_m
            if ascent:
                yp_ind = yp_ind + (self.config.inf_rate) * (g_m)
                # yp_ind = yp_ind + (self.config.inf_rate / math.sqrt((1+it))) * (g+noise)
            else:
                yp_ind = yp_ind - self.config.inf_rate * (g_m)

            if self.config.loglevel > 5:
                # yr = np.reshape(yp_ind, (-1, self.config.output_num, self.config.dimension))
                ypn = self.softmax(yp_ind, axis=-1)
                print(("energy:", np.average(e), "yind:", np.average(np.sum(np.square(yp_ind), 1)),
                      "gnorm:", np.average(gnorm), "yp:", np.average(np.max(ypn, 2))))

            # yp_a.append(np.reshape(yp, (-1, self.config.output_num* self.config.dimension)))
            yp_a.append(np.reshape(yp, (-1, self.config.output_num * self.config.dimension)))
            it += 1

        return np.array(yp_a)

    def search_inference(self, xinput=None, yinput=None, yinit=None, inf_iter=None, ascent=True, train=False):
        final_best = self.map_predict(xinput=xinput)

        # np.zeros((xinput.shape[0], self.config.output_num))
        found_point = np.zeros(xinput.shape[0])

        for iter in range(np.shape(xinput)[0]):
            random_proposal = np.copy(final_best[iter, :])
            yp_ind = np.reshape(self.var_to_indicator(np.expand_dims(random_proposal, 0)),
                                (-1, self.config.output_num * self.config.dimension))

            feed_dict = {
                self.x: np.expand_dims(xinput[iter], 0),
                self.yp_ind: yp_ind,
                self.margin_weight_ph: self.config.margin_weight,
                self.inf_penalty_weight_ph: self.config.inf_penalty,
                self.dropout_ph: self.config.dropout
            }

            score_first = self.sess.run(self.inf_objective, feed_dict=feed_dict)

            best_score = np.copy(score_first[:])
            labelset = set(np.arange(self.config.dimension))
            found = False
            random_proposal_new = np.copy(random_proposal[:])
            n = 0
            # while n < 50:
            for l in np.random.permutation(np.arange(self.config.output_num)):
                # l = random.randint(0,self.config.output_num-1)
                # if xtest[iter,l] == 0:
                #  break
                for label in labelset - set([random_proposal[l]]):
                    oldlabel = random_proposal_new[l]
                    random_proposal_new[l] = label
                    # score = self.evaluate(np.expand_dims(xtest[iter], 0),
                    #                      np.expand_dims(random_proposal_new, 0), yt=y)
                    yp_ind = np.reshape(self.var_to_indicator(np.expand_dims(random_proposal_new, 0)),
                                        (-1, self.config.output_num * self.config.dimension))
                    feed_dict = {
                        self.x: np.expand_dims(xinput[iter], 0),
                        self.yp_ind: yp_ind,
                        self.margin_weight_ph: self.config.margin_weight,
                        self.inf_penalty_weight_ph: self.config.inf_penalty,
                        self.dropout_ph: self.config.dropout
                    }
                    score = self.sess.run(self.inf_objective, feed_dict=feed_dict)

                    if score > best_score:
                        best_score = score

                        # best_l = l
                        # best_label = label
                        random_proposal_new[l] = label

                        if best_score > (score_first + self.config.score_margin) or best_score >= self.config.score_max:  #
                            found = True
                    else:
                        random_proposal_new[l] = oldlabel
                        # random_proposal[l] = random_proposal_new[l]
                        # changed = True
                        # break
                        if found:
                          break

                if found:
                    break
                n += 1
            found_point[iter] = 1 if found else 0

            if self.config.loglevel > 0:
                print(("inf, iter:", iter, "found:", found, "score first: ", score_first, "new score", best_score))

            final_best[iter, :] = np.copy(random_proposal_new)
            # if found:
            #  final_best[iter, best_l] = best_label
        return final_best

    def evaluate(self, xinput=None, yinput=None, yt=None):
        raise NotImplementedError



    def search_better_y_fast(self, xtest, yprev, yp, yt=None):
        final_best = np.zeros((xtest.shape[0], self.config.output_num))
        found_point = np.zeros(xtest.shape[0])

        for iter in range(np.shape(xtest)[0]):
            random_proposal = yprev[iter, :]
            if yt is not None:
                y = np.expand_dims(yt[iter], 0)
            else:
                y = None
            score_first = self.evaluate(np.expand_dims(xtest[iter], 0), np.expand_dims(random_proposal, 0), yt=y)
            best_score = np.copy(score_first[:])
            labelset = set(np.arange(self.config.dimension))
            found = False
            random_proposal_new = np.copy(random_proposal[:])
            n = 0
            #if score_first >= self.config.score_max:

            # while n < 50:
            best_distance = 0.0
            for l in np.random.permutation(np.arange(self.config.output_num)):
            #for l in np.arange(self.config.output_num):
                # l = random.randint(0,self.config.output_num-1)
                # if xtest[iter,l] == 0:
                #  break
                for label in (labelset - set([yprev[iter, l]])):  # set([random_proposal[l]]):


                    oldlabel = random_proposal_new[l]
                    random_proposal_new[l] = label
                    yprop = np.reshape(self.var_to_indicator(np.array([random_proposal_new])),
                                       (self.config.output_num * self.config.dimension))
                    ycurr = np.reshape(yp[iter, :], self.config.output_num * self.config.dimension)
                    # print yprop.shape, ycurr.shape
                    distance = np.linalg.norm(yprop - ycurr)

                    # if distance > self.config.distance_max:
                    #  continue
                    score = self.evaluate(np.expand_dims(xtest[iter], 0),
                                          np.expand_dims(random_proposal_new, 0), yt=y)

                    if self.config.loglevel > 60:
                        print((iter, l, distance, score, score_first))

                    if score > best_score:
                        best_score = score

                        # best_l = l
                        # best_label = label
                        random_proposal_new[l] = label

                        if best_score > (
                            score_first + self.config.score_margin) or best_score >= self.config.score_max:  #
                            #  or (best_score + self.config.score_margin > self.config.score_max ):
                            best_distance = distance
                            found = True
                    else:
                        random_proposal_new[l] = oldlabel
                        # random_proposal[l] = random_proposal_new[l]
                        # changed = True
                        # break
                    if found:
                        break

                if found:
                    break
                n += 1
            if best_score > score_first:
                found = True
            found_point[iter] = 1 if found else 0
            if self.config.loglevel > 40:
                print(("iter:", iter, "found:", found, "score first: ", score_first[0], "new score", best_score[0],
                      "dist", best_distance, "checked: ", l))

            final_best[iter, :] = np.copy(random_proposal_new)
            # if found:
            #  final_best[iter, best_l] = best_label
        return final_best, found_point

    def get_training_points(self, xinput=None, yinput=None, yinit=None, inf_iter=None, ascent=True):
        self.inf_objective = self.energy_yp
        self.inf_gradient = self.energy_ygradient

        y_a = self.inference(xinput=xinput, yinit=yinit, train=True, ascent=ascent, inf_iter=inf_iter)
        y_ans = y_a[-1]
        # y_ans = np.reshape(y_ans, (-1, self.config.output_num,self.config.dimension))
        # y_ans = np.argmax(y_ans, -1)

        # y_ans = self.var_to_indicator(y_ans)
        # y_ans = np.reshape(y_ans, (-1, self.config.output_num*self.config.dimension))

        # yp = self.sess.run(self.yp_h, feed_dict={self.yp_h_ind: y_ans})
        yp = np.reshape(y_ans, (-1, self.config.output_num, self.config.dimension))

        # yp = self.softmax(y_ans, axis=-1)
        # en_a = np.array([self.sess.run(self.inf_objective,
        #                               feed_dict={self.x: xinput,
        #                                          self.yp_ind: np.reshape(y_i, (
        #                                            -1, self.config.output_num * self.config.dimension)),
        #                                          self.inf_penalty_weight_ph: self.config.inf_penalty,
        #                                          self.dropout_ph: self.config.dropout})
        #                 for y_i in y_a])
        # ind = np.argmax(en_a, 0) if ascent else np.argmin(en_a, 0)
        # yp_ind = np.array([y_a[ind[i], i, :] for i in range(np.shape(xinput)[0])])
        # en_p = [en_a[ind[i], i] for i in range(np.shape(xinput)[0])]




        if self.config.use_search:
            y_better, found = self.search_better_y_fast(xinput, np.argmax(yp, 2), yp, yt=yinput)
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
                                                          self.dropout_ph: self.config.dropout}))
            en_p = np.array(self.sess.run(self.inf_objective,
                                          feed_dict={self.x: xinput,
                                                     self.yp_ind: np.reshape(y_ans, (
                                                         -1, self.config.output_num * self.config.dimension)),
                                                     self.inf_penalty_weight_ph: self.config.inf_penalty,

                                                     self.dropout_ph: self.config.dropout}))
        # y_a = np.array([yp, y_better])
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
                    print((i, fp[0], fb[0], en_p[i], en_better[i]))

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


        y_a = y_a[-10:]


        en_a = np.array([self.sess.run(self.inf_objective,
                                       feed_dict={self.x: xinput,
                                                  self.yp_ind: np.reshape(y_i, (
                                                      -1, self.config.output_num * self.config.dimension)),
                                                  self.inf_penalty_weight_ph: self.config.inf_penalty,
                                                  self.dropout_ph: self.config.dropout})
                         for y_i in y_a])
        # ind = np.argmax(en_a, 0) if ascent else np.argmin(en_a, 0)
        # yp_ind = np.array([y_a[ind[i], i, :] for i in range(np.shape(xinput)[0])])
        # en_p = [en_a[ind[i], i] for i in range(np.shape(xinput)[0])]
        # yp = self.softmax(yp_ind, axis=-1)




        # if np.random.random() > 0.0:
        #     print("search")
        #     y_better, found = self.search_better_y_fast(xinput, np.argmax(yp, 2), yp, yt=yinput)
        #     y_better = self.var_to_indicator(y_better)
        #
        #     y_a = np.array([yp, y_better])
        #
        #     # y_a = y_a[-4:]
        #
        #     en_better = np.array(self.sess.run(self.inf_objective,
        #                                        feed_dict={self.x: xinput,
        #                                                   self.yp_ind: np.reshape(y_better, (
        #                                                       -1, self.config.output_num * self.config.dimension)),
        #                                                   self.inf_penalty_weight_ph: self.config.inf_penalty,
        #                                                   self.dropout_ph: self.config.dropout}))
        #     print
        #     np.shape(en_better)
        #     en_a = np.array([en_p, en_better])

        # y_a = y_a[-2:]
        # en_a = en_a[-2:]
        f_a = np.array([self.evaluate(xinput=xinput, yinput=np.argmax(np.reshape(y_i, (-1, self.config.output_num, self.config.dimension)),2), yt=yinput) for y_i in y_a])
        # f_a = np.array([self.evaluate(xinput=xinput, yinput=np.argmax(y_i, 2), yt=yinput) for y_i in y_a])

        # print (np.average(en_a, axis=1))
        # print (np.average(f_a, axis=1))
        # y1 = np.argmax(y_a[-2],2)
        # y2 = np.argmax(y_a[-1],2)
        if self.config.loglevel > 25:
            for t in range(xinput.shape[0]):
                print((t, f_a[-2][t], f_a[-1][t], en_a[-2][t], en_a[-1][t]))

        size = np.shape(xinput)[0]
        t = np.array(list(range(size)))
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

    def get_first_large_consecutive_diff_old(self, xinput=None, yt=None, inf_iter=None, ascent=True):
        self.inf_objective = self.energy_yp
        self.inf_gradient = self.energy_ygradient

        y_a = self.inference(xinput=xinput, train=True, ascent=ascent, inf_iter=inf_iter)

        y_a = y_a[:]

        en_a = np.array([self.sess.run(self.inf_objective,
                                       feed_dict={self.x: xinput,
                                                  self.yp_ind: np.reshape(y_i, (
                                                  -1, self.config.output_num * self.config.dimension)),
                                                  self.inf_penalty_weight_ph: self.config.inf_penalty,
                                                  self.dropout_ph: self.config.dropout})
                         for y_i in y_a])
        f_a = np.array([self.evaluate(xinput=xinput, yinput=np.argmax(y_i, 2), yt=yt) for y_i in y_a])
        yp = y_a[-1]
        if self.config.loglevel > 4:
            for t in range(xinput.shape[0]):
                print((t, f_a[-2][t], f_a[-1][t], np.argmax(yp, 2)[t][:10]))

                # print (np.average(en_a, axis=1))
                # print (np.average(f_a, axis=1))

        size = np.shape(xinput)[0]
        t = np.array(list(range(size)))
        f1 = []
        f2 = []
        y1 = []
        y2 = []
        x = []
        k = 0
        it = np.shape(y_a)[0]
        for k in range(it - 1):
            for i in t:
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
                if violation > 0:
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
        # yp_a = np.array([self.softmax(yp) for yp in y_a])

        yp_a = np.array([self.sess.run(self.yp_h, feed_dict={self.yp_h_ind: y_i}) for y_i in y_a])
        # yp = np.reshape(yp, (-1, self.config.output_num, self.config.dimension))

        en_a = np.array([self.sess.run(self.inf_objective,
                                       feed_dict={self.x: xinput,
                                                  self.yp_ind: np.reshape(y_i, (
                                                  -1, self.config.output_num * self.config.dimension)),
                                                  self.inf_penalty_weight_ph: self.config.inf_penalty,
                                                  self.dropout_ph: self.config.dropout})
                         for y_i in yp_a])


        # ce_a = np.array(
        #     [np.sum(yinput * np.log(1e-20 + np.reshape(y_p, (-1, self.config.output_num * self.config.dimension))), 1)
        #      for y_p in yp_a])
        # print yinput.shape

        f_a = np.array([self.evaluate(xinput=xinput, yinput=np.argmax(np.reshape(y_i, (-1, self.config.output_num, self.config.dimension)),2), yt=np.argmax(np.reshape(yinput, (-1, self.config.output_num, self.config.dimension)),2)) for y_i in yp_a])

        e_t = self.sess.run(self.inf_objective,
                            feed_dict={self.x: xinput,
                                       self.yp_ind: np.reshape(yinput, (
                                           -1, self.config.output_num * self.config.dimension)),
                                       self.inf_penalty_weight_ph: self.config.inf_penalty,
                                       self.dropout_ph: self.config.dropout})
        #
        # print(np.average(en_a, axis=1))
        # print(np.average(ce_a, axis=1))

        size = np.shape(xinput)[0]
        t = np.array(list(range(size)))
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

    def h_predict(self, xinput=None, train=False, inf_iter=None, ascent=True):
        tflearn.is_training(is_training=train, session=self.sess)
        h_init = np.random.normal(0, 1, size=(np.shape(xinput)[0], self.config.hidden_num))
        feeddic = {self.x: xinput,
                   self.h: h_init,
                   self.inf_penalty_weight_ph: self.config.inf_penalty,
                   self.is_training: 1.0 if train else 0.0,
                   self.dropout_ph: self.config.dropout}
        h = self.sess.run(self.h_state, feed_dict=feeddic)
        return h

    def h_trajectory(self, xinput=None, train=False, inf_iter=None, ascent=True):
        tflearn.is_training(is_training=train, session=self.sess)
        h_init = np.random.normal(0, 1, size=(np.shape(xinput)[0], self.config.hidden_num))
        feeddic = {self.x: xinput,
                   self.h: h_init,
                   self.inf_penalty_weight_ph: self.config.inf_penalty,
                   self.is_training: 1.0 if train else 0.0,
                   self.dropout_ph: self.config.dropout}
        h_ar = self.sess.run(self.h_ar, feed_dict=feeddic)
        return h_ar

    def debug_predict(self, xinput=None, yinit=None, train=False, inf_iter=None, ascent=True, end2end=False):
        tflearn.is_training(is_training=train, session=self.sess)
        self.inf_objective = self.energy_yp
        self.inf_gradient = self.energy_ygradient
        y_a = self.inference(xinput=xinput, yinit=yinit, inf_iter=inf_iter, train=train, ascent=ascent)

        en, logits = self.sess.run([self.energy_yp, self.logits],
                                   feed_dict={self.x: xinput,
                                              self.yp_ind: np.reshape(y_a[-1], (
                                                  -1, self.config.output_num * self.config.dimension)),
                                              self.inf_penalty_weight_ph: self.config.inf_penalty,
                                              self.dropout_ph: self.config.dropout})
        for t in range(xinput.shape[0]):
            print(("t = " + str(t) + " energy: ", str(en[t])))
            print(("logits ", np.sum(logits[t, :]), logits[t, 1020:1025]))
            print(("y ", np.sum(y_a[-1, t, :]), y_a[-1, t, 1020:1025]))

    def soft_predict(self, xinput=None, yinit=None, train=False, inf_iter=None, ascent=True, end2end=False):
        tflearn.is_training(is_training=train, session=self.sess)
        if end2end:
            # h_init = np.random.normal(0, 1, size=(np.shape(xinput)[0], self.config.hidden_num))
            h_init = np.random.normal(0, 1, size=(np.shape(xinput)[0], self.config.hidden_num))
            feeddic = {self.x: xinput,
                       self.h: h_init,
                       self.is_training: 1.0 if train else 0.0,
                       self.inf_penalty_weight_ph: self.config.inf_penalty,
                       self.dropout_ph: self.config.dropout}
            yp = self.sess.run(self.yp, feed_dict=feeddic)
        else:

            self.inf_objective = self.energy_yp
            self.inf_gradient = self.energy_ygradient
            y_a = self.inference(xinput=xinput, yinit=yinit, inf_iter=inf_iter, train=train, ascent=ascent)

            # en_a = np.array([self.sess.run(self.energy_yp,
            #                   feed_dict={self.x: xinput,
            #                              self.yp_ind: np.reshape(y_i, (-1, self.config.output_num * self.config.dimension)),
            #                              self.inf_penalty_weight_ph: self.config.inf_penalty,
            #                              self.dropout_ph: self.config.dropout}) for y_i in y_a])
            # #try:
            # if self.config.loglevel > 5:
            #       for i in range(len(en_a)):
            #         y_i = y_a[i]
            #         f_i = np.array(self.evaluate(xinput=xinput, yinput=np.argmax(y_i, 2)))
            #         print ("----------------------------")
            #         print (i, np.average(en_a[i,:]), np.average(f_i))

            # y_ans = y_a[-1]

            # f_a = np.array([self.evaluate(xinput=xinput, yinput=np.argmax(y_i,2)) for y_i in y_a])
            # for t in range(xinput.shape[0]):
            #  print ("t = " + str(t)  + " energy: " + str(en_a[-1,t]))
            # ind = np.argmax(en_a,0) if ascent else np.argmin(en_a, 0)
            # y_ans = np.array([y_a[ind[i],i,:] for i in range(np.shape(xinput)[0])])

            # except:
            y_ans = y_a[-1]
            #yp = self.sess.run(self.yp_h, feed_dict={self.yp_h_ind: y_ans})
            yp = np.reshape(y_ans, (-1, self.config.output_num, self.config.dimension))
            #yp = self.softmax(y_ans, axis=-1)
            #yp = np.reshape(y_ans, (-1, self.config.output_num, self.config.dimension))

        return yp

    def map_predict_trajectory(self, xinput=None, train=False, inf_iter=None, ascent=True, end2end=False):
        if end2end:
            tflearn.is_training(train, self.sess)
            h_init = np.random.normal(0, 1, size=(np.shape(xinput)[0], self.config.hidden_num))
            feeddic = {self.x: xinput,
                       self.h: h_init,
                       self.inf_penalty_weight_ph: self.config.inf_penalty,
                       self.is_training: 1.0 if train else 0.0,
                       self.dropout_ph: self.config.dropout}
            soft_yp_ar, en_ar = self.sess.run([self.yp_ar, self.en_ar], feed_dict=feeddic)
            if self.config.dimension > 1:
                yp_ar = [np.argmax(yp, 2) for yp in soft_yp_ar]
            else:
                if self.config.verbose > 3:
                    for k in range(self.config.inf_iter):
                        print((np.average(en_ar[k])))
                yp_ar = soft_yp_ar
            return yp_ar
        else:
            raise NotImplementedError

    def map_predict_h(self, xinput=None, hidden=None):
        tflearn.is_training(False, self.sess)
        feeddic = {self.x: xinput,
                   self.h: hidden,
                   self.dropout_ph: self.config.dropout}
        yp = self.sess.run(self.yp_h, feed_dict=feeddic)
        return np.argmax(yp, 2)

    def map_predict(self, xinput=None, yinit=None, train=False, inf_iter=None, ascent=True, end2end=False,
                    continuous=False):
        yp = self.soft_predict(xinput=xinput, yinit=yinit, train=train, inf_iter=inf_iter, ascent=ascent,
                               end2end=end2end)
        if self.config.dimension == 1:
            return np.squeeze(yp)
        else:
            return np.argmax(yp, 2)

    # def inference_trajectory(self):

    def loss_augmented_soft_predict(self, xinput=None, yinput=None, yinit=None, train=False, inf_iter=None,
                                    ascent=True):
        self.inf_objective = self.loss_augmented_energy
        self.inf_gradient = self.loss_augmented_energy_ygradient
        y_a = self.inference(xinput=xinput, yinput=yinput, yinit=yinit, inf_iter=inf_iter, train=train, ascent=ascent)
        #
        en_a = np.array([self.sess.run(self.inf_objective,
                                       feed_dict={self.x: xinput,
                                                  self.yp_ind: np.reshape(ind_i, (
                                                  -1, self.config.output_num * self.config.dimension)),
                                                  self.yt_ind: yinput,
                                                  self.margin_weight_ph: self.config.margin_weight,
                                                  self.inf_penalty_weight_ph: self.config.inf_penalty,
                                                  self.dropout_ph: self.config.dropout}) for ind_i in y_a])

        # print ("en:", en_a[:,0])
        ind = np.argmax(en_a, 0) if ascent else np.argmin(en_a, 0)
        y_ans = np.array([y_a[ind[i], i, :] for i in range(np.shape(xinput)[0])])
        yp = self.softmax(y_ans, axis=-1)
        return yp

    def get_adverserial_predict(self, xinput=None, yinput=None, train=False, inf_iter=None, ascent=True):
        self.inf_objective = self.energy_yp
        self.inf_gradient = self.energy_ygradient
        yp_a = self.inference(xinput=xinput, yinput=yinput, inf_iter=inf_iter, train=train, ascent=ascent)
        yp_a = np.array([self.softmax(yp) for yp in yp_a])
        en_a = np.array([self.sess.run(self.inf_objective,
                                       feed_dict={self.x: xinput,
                                                  self.yp_ind: np.reshape(ind_i, (
                                                  -1, self.config.output_num * self.config.dimension)),
                                                  self.yt_ind: yinput,
                                                  self.margin_weight_ph: self.config.margin_weight,
                                                  self.inf_penalty_weight_ph: self.config.inf_penalty,
                                                  self.dropout_ph: self.config.dropout}) for ind_i in yp_a])

        ce_a = np.array(
            [-np.sum(yinput * np.log(1e-20 + np.reshape(y_p, (-1, self.config.output_num * self.config.dimension))), 1)
             for y_p in yp_a])
        print(("en:", np.average(en_a, axis=1), "ce:", np.average(ce_a, axis=1)))

        return self.softmax(yp_a[-1], axis=2, theta=1)

    def loss_augmented_map_predict(self, xd, train=False, inf_iter=None, ascent=True):
        yp = self.loss_augmented_soft_predict(xd, train=train, inf_iter=inf_iter, ascent=ascent)
        return np.argmax(yp, 2)

    def train_batch(self, xbatch=None, ybatch=None, verbose=0):
        raise NotImplementedError

    def train_unsupervised_rb_batch(self, xbatch=None, ybatch=None, yinit=None, verbose=0):
        tflearn.is_training(True, self.sess)
        it = 0

        while it < 1:
            it += 1

            x_b, y_h, y_l, l_h, l_l = self.get_first_large_consecutive_diff(xinput=xbatch, yinput=ybatch, ascent=True)
            dist = np.linalg.norm(np.reshape(y_h, y_l.shape) - y_l)
            total = np.size(l_h)
            indices = np.arange(0, total)
            # for b in range(total/bs + 1 ):
            if l_l.shape[0] <= 0:
                print()
                "skip"
                return

            # perm = np.random.permutation(range(total))
            #  indices = perm[b * bs:(b + 1) * bs]
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
                           self.margin_weight_ph: self.config.margin_weight})

            if verbose > 0:
                # print("************************************************************************************************")
                print((self.train_iter, o1, g, v1, v2, e1, e2, dist, np.shape(xbatch)[0], np.shape(x_b)[0],
                      np.average(l_l)))
        return


    def train_unsupervised_sg_batch(self, xbatch=None, ybatch=None, yinit=None, verbose=0):
        tflearn.is_training(True, self.sess)

        bs = 50

        n1 = 100000.0
        it = 0

        while it < 1:
            it += 1

            x_b, y_h, y_l, l_h, l_l = self.get_training_points(xinput=xbatch, yinput=ybatch, yinit=yinit,
                                                          ascent=True)
            dist = np.linalg.norm(np.reshape(y_h, y_l.shape) - y_l)
            total = np.size(l_h)
            indices = np.arange(0, total)
            # for b in range(total/bs + 1 ):
            if l_l.shape[0] <= 0:
                print("skip")
                return

            #  perm = np.random.permutation(range(total))
            #  indices = perm[b * bs:(b + 1) * bs]
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
                           self.margin_weight_ph: self.config.margin_weight})

            if verbose > 0:

                # print("************************************************************************************************")
                print((self.train_iter, o1, g, v1, v2, e1, e2, dist, np.shape(xbatch)[0], np.shape(x_b)[0], np.average(l_l)))
        return


    def train_supervised_cll_batch(self, xbatch=None, ybatch=None, yinit=None, verbose=0):
        tflearn.is_training(True, self.sess)
        #self.objective = 1

        feed_dict = {self.x: xbatch,
             self.yp_h_ind: np.reshape(ybatch, (-1, self.config.output_num * self.config.dimension)),
             self.learning_rate_ph: self.config.learning_rate,
             self.dropout_ph: self.config.dropout,
             self.inf_penalty_weight_ph: self.config.inf_penalty,
             self.margin_weight_ph: self.config.margin_weight}

        _, o = self.sess.run([self.train_step, self.objective], feed_dict=feed_dict)


        if verbose > 0:
            print((self.train_iter, o ))
        return

    def train_supervised_batch(self, xbatch, ybatch, yinit=None, verbose=0):
        tflearn.is_training(True, self.sess)

        if self.config.dimension > 1:
            yt_ind = self.var_to_indicator(ybatch)
            yt_ind = np.reshape(yt_ind, (-1, self.config.output_num * self.config.dimension))
        else:
            yt_ind = ybatch
        # xd, yd, yp_ind = self.get_all_diff(xinput=xbatch, yinput=yt_ind, ascent=True, inf_iter=10)
        yp_ind = self.loss_augmented_soft_predict(xinput=xbatch, yinput=yt_ind, yinit=yinit, train=True, ascent=True)
        # yp_ind = self.soft_predict(xinput=xbatch, yinit=yinit, train=True, ascent=True)
        yp_ind = np.reshape(yp_ind, (-1, self.config.output_num * self.config.dimension))
        # yt_ind = np.reshape(yd, (-1, self.config.output_num*self.config.dimension))

        feeddic = {self.x: xbatch, self.yp_ind: yp_ind, self.yt_ind: yt_ind,
                   self.learning_rate_ph: self.config.learning_rate,
                   self.margin_weight_ph: self.config.margin_weight,
                   self.inf_penalty_weight_ph: self.config.inf_penalty,
                   self.dropout_ph: self.config.dropout}

        _, o, ce, n, en_yt, en_yhat = self.sess.run(
            [self.train_step, self.objective, self.ce, self.num_update, self.total_energy_yt, self.total_energy_yp],
            feed_dict=feeddic)
        if verbose > 0:
            print((self.train_iter, o, n, en_yt, en_yhat))
        return n

    def train_supervised_value_batch(self, xbatch, ybatch, yinit=None, verbose=0):
        tflearn.is_training(True, self.sess)

        if self.config.dimension > 1:
            yt_ind = self.var_to_indicator(ybatch)
            yt_ind = np.reshape(yt_ind, (-1, self.config.output_num * self.config.dimension))
        else:
            yt_ind = ybatch

        if random.uniform(0, 1) > 0.3:
            yp = self.soft_predict(xinput=xbatch, yinit=yinit, ascent=True)
            # yp_ind = self.var_to_indicator(yp)
            yp_ind = np.reshape(yp, (-1, self.config.output_num * self.config.dimension))
            if self.config.dimension > 1:
                v = self.evaluate(xinput=xbatch, yinput=np.argmax(yp), yt=ybatch)
                # print v
            else:
                v = self.evaluate(xinput=xbatch, yinput=yp, yt=ybatch)

        else:
            v = self.evaluate(xinput=xbatch, yinput=ybatch, yt=ybatch)
            yp_ind = yt_ind[:]

        feeddic = {self.x: xbatch, self.yp_ind: yp_ind, self.v_ind: v,
                   self.learning_rate_ph: self.config.learning_rate,
                   self.margin_weight_ph: self.config.margin_weight,
                   self.inf_penalty_weight_ph: self.config.inf_penalty,
                   self.dropout_ph: self.config.dropout}

        _, o, ce, en = self.sess.run([self.train_step, self.objective, self.newce, self.en], feed_dict=feeddic)
        if verbose > 0:
            print((self.train_iter, o, ce, np.sum(v), en))
        return 1

    def train_supervised_e2e_batch(self, xbatch, ybatch, verbose=0):
        tflearn.is_training(True, self.sess)
        if self.config.dimension > 1:
            yt_ind = self.var_to_indicator(ybatch)
            yt_ind = np.reshape(yt_ind, (-1, self.config.output_num * self.config.dimension))
        else:
            yt_ind = ybatch

        h_init = np.random.normal(0, 1, size=(np.shape(xbatch)[0], self.config.hidden_num))
        # h_0 = np.zeros((np.shape(xbatch)[0], self.config.hidden_num))
        feeddic = {self.x: xbatch, self.yt_ind: yt_ind,
                   self.h: h_init,
                   # self.h0 : h_0,
                   self.learning_rate_ph: self.config.learning_rate,
                   self.inf_penalty_weight_ph: self.config.inf_penalty,
                   self.is_training: 1.0,
                   self.dropout_ph: self.config.dropout}

        if self.train_iter % 2 == 0:  # < self.config.pretrain_iter:
            _, o = self.sess.run([self.train_pred_step, self.objective], feed_dict=feeddic)

        else:
            if self.config.pretrain_iter < 0:
                _, o, en_ar, g_ar, h_ar = self.sess.run(
                    [self.train_all_step, self.objective, self.en_ar, self.g_ar, self.h_ar], feed_dict=feeddic)

            else:
                _, o, en_ar, g_ar, h_ar = self.sess.run(
                    [self.train_step, self.objective, self.en_ar, self.g_ar, self.h_ar], feed_dict=feeddic)

            if verbose > 0:
                print("---------------------------------------------------------")
                for k in range(self.config.inf_iter):
                    print((np.average(np.linalg.norm(g_ar[k], axis=1)), np.average(np.linalg.norm(h_ar[k], axis=1)),
                          np.average(en_ar[k]), ))
        return o

    def train_supervised_e2e_batch2(self, xbatch, ybatch, verbose=0):
        tflearn.is_training(True, self.sess)
        if self.config.dimension > 1:
            yt_ind = self.var_to_indicator(ybatch)
            yt_ind = np.reshape(yt_ind, (-1, self.config.output_num * self.config.dimension))
        else:
            yt_ind = ybatch

        h_init = np.random.normal(0, 1, size=(np.shape(xbatch)[0], self.config.hidden_num))
        # h_0 = np.zeros((np.shape(xbatch)[0], self.config.hidden_num))
        feeddic = {self.x: xbatch, self.yt_ind: yt_ind,
                   self.h: h_init,
                   # self.h0 : h_0,
                   self.learning_rate_ph: self.config.learning_rate,
                   self.inf_penalty_weight_ph: self.config.inf_penalty,
                   self.is_training: 1.0,
                   self.dropout_ph: self.config.dropout}

        if self.train_iter < self.config.pretrain_iter:
            _, o = self.sess.run([self.train_hspen_step, self.objective], feed_dict=feeddic)

        else:
            _, o = self.sess.run(
                [self.train_predhx_step, self.objective], feed_dict=feeddic)
        return o

    def save(self, path):
        self.saver.save(self.sess, path)

    def restore(self, path):
        self.saver.restore(self.sess, path)

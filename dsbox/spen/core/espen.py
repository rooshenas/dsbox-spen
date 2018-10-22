import tensorflow as tf
import numpy as np

import tflearn
import tflearn.initializations as tfi
from enum import Enum
import math
import random

NEG_SAMPLES_PER_DATA_POINT = 5


class InfInit(Enum):
    Random_Initialization = 1
    GT_Initialization = 2
    Zero_Initialization = 3


class TrainingType(Enum):
    Value_Matching = 1
    SSVM = 2
    Rank_Based = 3
    Rank_Based_Expected = 4
    End2End = 5
    CLL = 6


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

    def rank_based_training_expected(self):

        # feed_dict = {self.x: x_b,  # (600, 1836) --> bs, nf
        #              self.yp_h_ind: np.reshape(y_h, (-1, self.config.output_num * self.config.dimension)),
        #              # (600, 318) --> bs, output_num (159) x dimension i.e. num_labels x label_dimension
        #              self.yp_l_ind: np.reshape(y_l, (-1, self.config.output_num * self.config.dimension)),
        #              # (600, 318) --> bs, output_num (159) x dimension
        #              self.value_l: l_l,  # bs -- task loss value for lower configuration
        #              self.value_h: l_h,  # bs -- task loss value for higher configuration
        #              self.learning_rate_ph: self.config.learning_rate,  # learning rate : scalar
        #              self.dropout_ph: self.config.dropout,  # dropout rate: scalar
        #              self.inf_penalty_weight_ph: self.config.inf_penalty,
        #              # penalty : which is set to zero at the moment
        #              self.margin_weight_ph: self.config.margin_weight})  # margin_weight : also scalar
        #

        self.margin_weight_ph = tf.placeholder(tf.float32, shape=[], name="Margin")
        self.inf_penalty_weight_ph = tf.placeholder(tf.float32, shape=[], name="InfPenalty")
        self.yp_h_ind = tf.placeholder(tf.float32,
                                       shape=[None, self.config.output_num * self.config.dimension],
                                       name="YP_H")

        self.yp_l_ind = tf.placeholder(tf.float32,
                                       shape=[None, self.config.output_num * self.config.dimension],
                                       name="YP_L")

        yp_ind_sm_h = tf.nn.softmax(tf.reshape(self.yp_h_ind, [-1, self.config.output_num, self.config.dimension]))
        # [batch_size x num_labels x softmax(label_dimension)]
        self.yp_h = tf.reshape(yp_ind_sm_h, [-1, self.config.output_num * self.config.dimension])
        # [[batch_size x num_labels] x softmax(label_dimension)]

        yp_ind_sm_l = tf.nn.softmax(tf.reshape(self.yp_l_ind, [-1, self.config.output_num, self.config.dimension]))
        # [batch_size x num_labels x softmax(label_dimension)]
        self.yp_l = tf.reshape(yp_ind_sm_l, [-1, self.config.output_num * self.config.dimension])
        # [[batch_size x num_labels] x softmax(label_dimension)]

        self.value_h = tf.placeholder(tf.float32, shape=[None])
        # Float scalar value
        self.value_l = tf.placeholder(tf.float32, shape=[None])
        # Float scalar value

        self.yh_penalty = self.inf_penalty_weight_ph * tf.reduce_sum(tf.square(self.yp_h_ind), 1)
        # scalar: inference penalty sum, we square along the label dimension and sum it

        self.yl_penalty = self.inf_penalty_weight_ph * tf.reduce_sum(tf.square(self.yp_l_ind), 1)
        # same as above

        self.energy_yh_ = self.get_energy(xinput=self.x, yinput=self.yp_h, embedding=self.embedding,
                                          reuse=False)  # - self.yh_penalty
        # inputs: self.x: bs x nf  and yp_h: bs, output_num (159) x dimension i.e. num_labels x label_dimension
        # output is a scalar

        self.energy_yl_ = self.get_energy(xinput=self.x, yinput=self.yp_l, embedding=self.embedding,
                                          reuse=True)  # - self.yl_penalty
        # save as previous

        self.energy_yh = self.energy_yh_ - self.yh_penalty
        self.energy_yl = self.energy_yl_ - self.yl_penalty

        self.yp_ind = self.yp_h_ind
        self.yp = self.yp_h
        self.energy_yp = self.energy_yh

        # self.en = -tf.reduce_sum(self.yp * tf.log( tf.maximum(self.yp, 1e-20)), 1)

        self.energy_ygradient = tf.gradients(self.energy_yp, self.yp_ind)[0]   # dimension ??? no clue

        self.ce = -tf.reduce_sum(self.yp_h * tf.log(tf.maximum(self.yp_l, 1e-20)), 1)   # [[batch_size x num_labels] x softmax(label_dimension)] -- cross entropy computation on the third dimension

        self.diff = (self.value_h - self.value_l) * self.margin_weight_ph       # difference between values

        self.objective = tf.reduce_sum(tf.maximum(-self.energy_yh + self.energy_yl + self.diff, 0.0)) \
                         + self.config.l2_penalty * self.get_l2_loss()  # L2 regularization with margin loss

        self.num_update = tf.reduce_sum(tf.cast((self.diff > (self.energy_yh - self.energy_yl)), tf.float32))   # tracks number of points that are useful for gradient updates

        self.vh_sum = tf.reduce_mean(self.value_h)      # takes mean over the batch
        self.vl_sum = tf.reduce_mean(self.value_l)      # takes mean over the batch
        self.eh_sum = tf.reduce_mean(self.energy_yh)    # takes mean over the batch
        self.el_sum = tf.reduce_mean(self.energy_yl)    # takes mean over the batch

        # self.train_step = self.optimizer.minimize(self.objective, var_list=self.energy_variables())
        # self.train_step = self.optimizer.minimize(self.objective)
        grads_vals = self.optimizer.compute_gradients(self.objective)

        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_vals]
        #train_op = optimizer.apply_gradients(capped_gvs)
        # self.grad_norm = tf.reduce_mean(tf.norm(grads, 1))

        self.grad_norm = tf.constant(0.0)   # don't know what is the purpose of this guy?
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
        elif training_type == TrainingType.Rank_Based_Expected:
            return self.rank_based_training_expected()
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


    def inference(self, xinput=None, yinput=None, yinit=None, inf_iter=None, ascent=True, train=False):

        if self.config.loglevel > 5:
            print("Inside inference: ...")

        if inf_iter is None:
            inf_iter = self.config.inf_iter # TODO: local variable inf_iter is never used

        tflearn.is_training(is_training=train, session=self.sess)

        size = list(np.shape(xinput))[0]

        # Pre-paring yt_ind (true labels) and yp_ind (initial predictions)
        # if yp_init is not provided then it is initialized randomly
        # Interestingly yt_ind is not utilized for rank based training

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

        i = 0
        yp_a = []
        g_m = np.zeros(np.shape(yp_ind))  # Gradient momentum, why not use inbuilt moment things?
        alpha = self.config.alpha
        mean = np.zeros(shape=np.shape(yp_ind))
        avg_gnorm = 100.0

        it = 0
        while avg_gnorm > self.config.gradient_thresh and it < self.config.inf_iter:

            # Forward pass and provide output yp
            feed_dict = {self.x: xinput, self.yp_ind: yp_ind,
                         self.margin_weight_ph: self.config.margin_weight,
                         self.inf_penalty_weight_ph: self.config.inf_penalty,
                         self.dropout_ph: self.config.dropout}
            yp = self.sess.run(self.yp_h, feed_dict=feed_dict)

            # Backward pass over the inference procedure
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

            # Checking gradients and making updates
            g = np.clip(g, a_min=-1.0, a_max=1.0)       # TODO: shouldn't we take this out and tune it?
            gnorm = np.linalg.norm(g, axis=1)
            avg_gnorm = np.average(gnorm)

            if train:
                noise = np.random.normal(mean, self.config.noise_rate * np.average(gnorm) / math.sqrt((1 + it)), size=np.shape(g))
            else:
                noise = np.zeros(shape=np.shape(g))

            # Gradient update -- alpha is a scalar for gradient momentum
            # No noise if not training
            g_m = alpha * (g + noise) + (1 - alpha) * g_m

            if ascent:
                yp_ind = yp_ind + self.config.inf_rate * g_m
            else:
                yp_ind = yp_ind - self.config.inf_rate * g_m

            if self.config.loglevel > 5:

                ypn = self.softmax(yp_ind, axis=-1)
                print("energy:", np.average(e), "yind:", np.average(np.sum(np.square(yp_ind), 1)),
                      "gnorm:", np.average(gnorm), "yp:", np.average(np.max(ypn, 2)))

            yp_a.append(np.reshape(yp, (-1, self.config.output_num * self.config.dimension)))
            it += 1

        return np.array(yp_a)


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
                        print(iter, l, distance, score, score_first)

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
                print("iter:", iter, "found:", found, "score first: ", score_first[0], "new score", best_score[0],
                      "dist", best_distance, "checked: ", l)

            final_best[iter, :] = np.copy(random_proposal_new)
            # if found:
            #  final_best[iter, best_l] = best_label
        return final_best, found_point

    def get_training_points(self, xinput=None, yinput=None, yinit=None, inf_iter=None, ascent=True):

        # TODO: Why this assignment is done here?
        self.inf_objective = self.energy_yp
        self.inf_gradient = self.energy_ygradient


        y_a = self.inference(xinput=xinput, yinit=yinit, train=True, ascent=ascent, inf_iter=inf_iter)
        y_ans = y_a[-1]
        y_ans = np.reshape(y_ans, (-1, self.config.output_num,self.config.dimension))
        y_ans = np.argmax(y_ans, -1)

        y_ans = self.var_to_indicator(y_ans)
        y_ans = np.reshape(y_ans, (-1, self.config.output_num*self.config.dimension))

        yp = self.sess.run(self.yp_h, feed_dict={self.yp_h_ind: y_ans})
        yp = np.reshape(yp, (-1, self.config.output_num, self.config.dimension))


        if self.config.use_search: # this step is not used for supervised training
            y_better, found = self.search_better_y_fast(xinput, np.argmax(yp, 2), yp, yt=yinput)
        else:
            y_better = yinput
            found = np.ones(yp.shape[0])
        yb = self.var_to_indicator(y_better)



        if self.config.loglevel > 30:   # TODO: Remove these magic numbers
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

        # TODO: Can we get rid of this loop? It should be possible as long as we can run self.evaluate in batch
        # What is the use of found? Is it relevant for my task.
        # Its running over batch samples
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
        fh = np.array(fh)       # Evaluation score for correct structure
        fl = np.array(fl)       # Evaluation score for incorrect structure
        yh = np.array(yh)       # Output configuration for correct structure
        yl = np.array(yl)       # Output configuration for incorrect structure

        return x, yh, yl, fh, fl


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

            y_ans = y_a[-1]
            yp = np.reshape(y_ans, (-1, self.config.output_num, self.config.dimension))

        return yp


    def map_predict(self, xinput=None, yinit=None, train=False, inf_iter=None, ascent=True, end2end=False,
                    continuous=False):
        yp = self.soft_predict(xinput=xinput, yinit=yinit, train=train, inf_iter=inf_iter, ascent=ascent,
                               end2end=end2end)
        if self.config.dimension == 1:
            return np.squeeze(yp)
        else:
            return np.argmax(yp, 2)


    def train_batch(self, xbatch=None, ybatch=None, verbose=0):
        raise NotImplementedError

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
                print "skip"
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
                print(self.train_iter, o1, g, v1, v2, e1, e2, dist, np.shape(xbatch)[0], np.shape(x_b)[0], np.average(l_l))
        return

    def get_more_samples_from_yp(self, prob_dist):

        # self.config.output_num -- label dimension
        # self.config.dimension -- number of labels

        cum_prob_dist = np.cumsum(prob_dist, axis=1)
        samples = np.random.uniform(0, 1, NEG_SAMPLES_PER_DATA_POINT * self.config.output_num).reshape(NEG_SAMPLES_PER_DATA_POINT, self.config.output_num)

        samples = np.where(samples[:, np.newaxis] - cum_prob_dist.T > 0, 0, 1).swapaxes(1, 2).reshape(NEG_SAMPLES_PER_DATA_POINT *self.config.output_num, self.config.dimension)
        yp_samples =  np.zeros_like(samples)
        yp_samples[np.arange(samples.shape[0]), np.argmax(samples, axis=1)] = 1
        return yp_samples.reshape(NEG_SAMPLES_PER_DATA_POINT, self.config.output_num, self.config.dimension)

    def get_expected_prediction_samples(self, xinput=None, yinput=None, yinit=None, inf_iter=None, ascent=True):

        self.inf_objective = self.energy_yp
        self.inf_gradient = self.energy_ygradient

        y_a = self.inference(xinput=xinput, yinit=yinit, train=True, ascent=ascent, inf_iter=inf_iter)
        y_ans = y_a[-1]
        y_ans = np.reshape(y_ans, (-1, self.config.output_num, self.config.dimension))
        y_ans = np.argmax(y_ans, -1)

        y_ans = self.var_to_indicator(y_ans)
        y_ans = np.reshape(y_ans, (-1, self.config.output_num * self.config.dimension))

        yp = self.sess.run(self.yp_h, feed_dict={self.yp_h_ind: y_ans})
        yp = np.reshape(yp, (-1, self.config.output_num, self.config.dimension))

        if self.config.use_search:  # this step is not used for supervised training
            raise NotImplementedError
        else:
            y_better = yinput
            found = np.ones(yp.shape[0])
        yb = self.var_to_indicator(y_better)

        # y_a = np.array([yp, y_better])
        fh, fl, yh, yl, x = [], [], [], [], []

        # TODO: Can we get rid of this loop? It should be possible as long as we can run self.evaluate in batch
        # What is the use of found? Is it relevant for my task.
        # Its running over batch samples
        for i in range(yp.shape[0]):

            if found[i] > 0:
                if yinput is not None:

                    # Adding the current prediction and ground truth pair
                    fp = self.evaluate(xinput=xinput[i], yinput=np.expand_dims(np.argmax(yp[i], 1), 0),
                                       yt=np.expand_dims(yinput[i], 0))
                    fb = self.evaluate(xinput=xinput[i], yinput=np.expand_dims(y_better[i], 0),
                                       yt=np.expand_dims(yinput[i], 0))
                    fh.append(fb[0])
                    fl.append(fp[0])
                    yh.append(yb[i])

                    yl.append(y_ans[i])
                    x.append(xinput[i])

                    # Adding more negative samples
                    more_samples_from_yp = self.get_more_samples_from_yp(yp[i])
                    for sample_index in range(NEG_SAMPLES_PER_DATA_POINT):

                        # Adding sample
                        fp = self.evaluate(xinput=xinput[i], yinput=np.expand_dims(np.argmax(more_samples_from_yp[sample_index], 1), 0), yt=np.expand_dims(yinput[i], 0))
                        yl.append(more_samples_from_yp[sample_index].flatten())

                        # Adding corresponding ground truth and input
                        fh.append(fb[0])
                        fl.append(fp[0])
                        yh.append(yb[i])
                        x.append(xinput[i])


                else:
                    raise NotImplementedError


        x = np.array(x)
        fh = np.array(fh)  # Evaluation score for correct structure
        fl = np.array(fl)  # Evaluation score for incorrect structure
        yh = np.array(yh)  # Output configuration for correct structure
        yl = np.array(yl)  # Output configuration for incorrect structure

        return x, yh, yl, fh, fl

    def train_expected_supervised_batch(self, xbatch=None, ybatch=None, yinit=None, verbose=0):
        tflearn.is_training(True, self.sess)

        bs = 50

        n1 = 100000.0
        it = 0

        while it < 1:
            it += 1

            x_b, y_h, y_l, l_h, l_l = self.get_expected_prediction_samples(xinput=xbatch, yinput=ybatch, yinit=yinit,
                                                               ascent=True)
            dist = np.linalg.norm(np.reshape(y_h, y_l.shape) - y_l)
            total = np.size(l_h)
            indices = np.arange(0, total)
            # for b in range(total/bs + 1 ):
            if l_l.shape[0] <= 0:
                print("skip")
                return

            # perm = np.random.permutation(range(total))
            #  indices = perm[b * bs:(b + 1) * bs]
            _, o1, g, n1, v1, v2, e1, e2 = self.sess.run(
                [self.train_step, self.objective, self.grad_norm, self.num_update, self.vh_sum, self.vl_sum,
                 self.eh_sum, self.el_sum],
                feed_dict={self.x: x_b, # (600, 1836) --> bs, nf
                           self.yp_h_ind: np.reshape(y_h, (-1, self.config.output_num * self.config.dimension)), # (600, 318) --> bs, output_num (159) x dimension i.e. num_labels x label_dimension
                           self.yp_l_ind: np.reshape(y_l, (-1, self.config.output_num * self.config.dimension)), # (600, 318) --> bs, output_num (159) x dimension
                           self.value_l: l_l, # bs -- task loss value for lower configuration
                           self.value_h: l_h, # bs -- task loss value for higher configuration
                           self.learning_rate_ph: self.config.learning_rate,  # learning rate : scalar
                           self.dropout_ph: self.config.dropout, # dropout rate: scalar
                           self.inf_penalty_weight_ph: self.config.inf_penalty, # penalty : which is set to zero at the moment
                           self.margin_weight_ph: self.config.margin_weight})   # margin_weight : also scalar

            if verbose > 0:
                # print("************************************************************************************************")
                print(self.train_iter, o1, g, v1, v2, e1, e2, dist, np.shape(xbatch)[0], np.shape(x_b)[0],
                      np.average(l_l))
        return

    def save(self, path):
        self.saver.save(self.sess, path)

    def restore(self, path):
        self.saver.restore(self.sess, path)

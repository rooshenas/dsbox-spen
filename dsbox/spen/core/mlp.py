import tensorflow as tf
import tflearn
import tflearn.initializations as tfi
import numpy as np


class MLP:
    def __init__(self, config):
        self.config = config
        self.x = tf.placeholder(tf.float32, shape=[None, self.config.input_num], name="InputX")
        self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name="LearningRate")
        self.dropout_ph = tf.placeholder(tf.float32, shape=[], name="Dropout")
        self.embedding = None
        self.y = tf.placeholder(tf.float32, shape=[None, config.output_num * self.config.dimension], name="OutputX")
        self.labels = tf.placeholder(tf.float32, shape=[None, config.output_num], name="labels")

    def init_embedding(self, embedding):
        self.sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: embedding})
        return self

    def init(self):
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)
        self.saver = tf.train.Saver()
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

    def get_loss(self, yt, yp):
        raise NotImplementedError

    def mse_loss(self, yt, yp):
        l = tf.reduce_mean(tf.square(yt - yp) * 255.0)
        return l

    def biased_loss(self, yt, yp):
        l = -tf.reduce_sum((tf.reduce_sum(yt * tf.log(tf.maximum(yp, 1e-20)), 1) \
                            + tf.reduce_sum((1. - yt) * tf.log(tf.maximum(1. - yp, 1e-20)), 1)))
        yp = tf.reshape(yp, [-1, self.config.output_num, self.config.dimension])
        yp_zeros = yp[:, :, 0]
        yp_ones = yp[:, :, 1]
        return l + 1.2 * (tf.reduce_sum(yp_zeros) - tf.reduce_sum(yp_ones))

    def get_feature_net_mlp(self, xinput, output_num, reuse=False):
        print(output_num)

        net = xinput
        j = 0
        for (sz, a) in self.config.layer_info:
            print(sz, a)
            net = tflearn.fully_connected(net, sz,
                                          weight_decay=self.config.weight_decay,
                                          # weights_init=tfi.variance_scaling(,
                                          bias_init=tfi.zeros(), regularizer='L2', reuse=reuse, scope=("fx.h" + str(j)))
            net = tflearn.activations.relu(net)
            # net = tflearn.dropout(net, 1.0 - self.config.dropout)
            j = j + 1
        logits = tflearn.fully_connected(net, output_num,
                                         activation='linear',
                                         regularizer='L2',
                                         weight_decay=self.config.weight_decay,
                                         weights_init=tfi.variance_scaling(),
                                         bias=False,
                                         reuse=reuse, scope=("fx.h" + str(j)))
        return logits

    def ce_loss(self, yt, yp):
        l = -tf.reduce_sum((tf.reduce_sum(yt * tf.log(tf.maximum(yp, 1e-20)), 1) \
                            + tf.reduce_sum((1. - yt) * tf.log(tf.maximum(1. - yp, 1e-20)), 1)))
        return l

    def softmax_prediction_network2(self, xinput=None, reuse=False):
        net = xinput
        print("xinput", xinput)
        with tf.variable_scope("pred") as scope:
            net = tflearn.fully_connected(net, 1000, regularizer='L2',
                                          weight_decay=self.config.weight_decay,
                                          weights_init=tfi.variance_scaling(),
                                          bias_init=tfi.zeros(), reuse=reuse,
                                          scope=("ph.0"))
            net = tf.nn.relu(net)
            net = tflearn.layers.dropout(net, 1 - self.config.dropout)

            net = tflearn.fully_connected(net, self.config.output_num * self.config.dimension, activation='linear',
                                          weight_decay=self.config.weight_decay,
                                          weights_init=tfi.variance_scaling(),
                                          bias_init=tfi.zeros(), reuse=reuse,
                                          regularizer='L2',
                                          scope=("ph.1"))

        cat_output = tf.reshape(net, (-1, self.config.output_num, self.config.dimension))

        return tf.nn.softmax(cat_output, dim=2)

    def mlp_prediction_network(self, xinput=None, reuse=False):
        net = xinput
        j = 0
        with tf.variable_scope("pred") as scope:
            for (sz, a) in self.config.pred_layer_info:
                print(sz, a)
                net = tflearn.fully_connected(net, sz, regularizer='L2', activation=a,
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              bias_init=tfi.zeros(), reuse=reuse,
                                              scope=("ph." + str(j)))
                net = tflearn.layers.dropout(net, 1 - self.config.dropout)
                j = j + 1

            logits = tflearn.fully_connected(net, self.config.output_num * self.config.dimension, activation='linear',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(),
                                             reuse=reuse,
                                             bias=False,
                                             regularizer='L2',
                                             scope=("ph.fc"))
        # if self.config.dimension == 1:
        #  return tf.nn.sigmoid(net)
        # else:
        #  cat_output = tf.reshape(net, (-1, self.config.output_num, self.config.dimension))
        #  return tf.nn.softmax(cat_output, dim=2)
        return logits

    def mlph_prediction_network(self, xinput=None, reuse=False):
        hpart = xinput[:, :self.config.hidden_num]
        xpart = xinput[:, self.config.hidden_num:]

        with tf.variable_scope("hpred") as scope:
            hnet = tflearn.fully_connected(hpart, 200, regularizer='L2', activation='relu',
                                           weight_decay=self.config.weight_decay,
                                           weights_init=tfi.zeros(),
                                           bias_init=tfi.zeros(), reuse=reuse,
                                           scope=("f.h0"))

        with tf.variable_scope("xpred") as scope:
            xnet = tflearn.fully_connected(xpart, 1024, regularizer='L2', activation='relu',
                                           weight_decay=self.config.weight_decay,
                                           weights_init=tfi.variance_scaling(),
                                           bias_init=tfi.zeros(), reuse=reuse,
                                           scope=("f.x0"))
        j = 0
        net = tf.concat((hnet, xnet), axis=1)
        with tf.variable_scope("pred") as scope:
            for (sz, a) in self.config.pred_layer_info:
                print(sz, a)
                net = tflearn.fully_connected(net, sz, regularizer='L2',
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              bias_init=tfi.zeros(), reuse=reuse,
                                              scope=("ph." + str(j)))
                net = tf.nn.relu(net)
                net = tflearn.layers.dropout(net, 1 - self.config.dropout)
                j = j + 1

            net = tflearn.fully_connected(net, self.config.output_num * self.config.dimension, activation='linear',
                                          weight_decay=self.config.weight_decay,
                                          weights_init=tfi.variance_scaling(),
                                          reuse=reuse,
                                          bias=False,
                                          regularizer='L2',
                                          scope=("ph.fc"))
        if self.config.dimension == 1:
            return tf.nn.sigmoid(net)
        else:
            cat_output = tf.reshape(net, (-1, self.config.output_num, self.config.dimension))
            return tf.nn.softmax(cat_output, dim=2)

    def horses_network(self, xinput=None, reuse=False):
        net = xinput
        j = 0
        with tf.variable_scope("pred"):
            net = tf.reshape(net, shape=(-1, self.config.image_width, self.config.image_height, 3))
            net = tf.reshape(net, shape=(-1, self.config.image_width, self.config.image_height, 3))
            net = tf.layers.conv2d(inputs=net, filters=64,
                                   kernel_size=[3, 3], strides=1, padding="same", name="conv0",
                                   activation=tf.nn.relu, reuse=reuse)

            net = tf.layers.conv2d(inputs=net, filters=64, strides=2,
                                   kernel_size=[3, 3], padding="same", name="conv1",
                                   activation=tf.nn.relu, reuse=reuse)
            #  net = tf.layers.batch_normalization()
            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool1")
            # net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            net = tf.layers.conv2d(inputs=net, filters=128,
                                   kernel_size=[3, 3], padding="same", name="conv2", strides=2,
                                   activation=tf.nn.relu, reuse=reuse)

            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool2")
            # net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            net = tf.layers.conv2d(inputs=net, filters=128,
                                   kernel_size=[3, 3 ], padding="same", name="conv3", strides=2,
                                   activation=tf.nn.relu, reuse=reuse)
            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool3")
            # net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            # net = tf.reshape(net, [-1, (self.config.image_width / 8) * (self.config.image_height / 8) * 128])

            net = tf.layers.flatten(net)
            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool3")
            net = tflearn.dropout(net, 1.0 - self.config.dropout)
            j = 0
            for (sz, a) in self.config.layer_info:
                net = tflearn.fully_connected(net, sz,
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              activation=a,
                                              bias_init=tfi.zeros(), regularizer='L2', reuse=reuse,
                                              scope=("fx.h" + str(j)))
                net = tflearn.dropout(net, 1.0 - self.config.dropout)

                # net = tflearn.layers.dropout(net, 1 - self.config.dropout)
                j = j + 1

            logits = tflearn.fully_connected(net, self.config.output_num * self.config.dimension, activation='linear',
                                             regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(), bias=False,
                                             reuse=reuse, scope=("fx.fc"))
        return logits

    def cnn_javier(self, xinput=None, reuse=False):
        net = tf.reshape(xinput, (-1, 32, 64, 1))
        prev_nFilter = 0
        with tf.variable_scope("pred"):
            for j, (nFilter, kSz, strides) in enumerate(self.config.cnn_layer_info):
                if j == 0:
                    net = tflearn.conv_2d(net, 1, 3, reuse=False, scope="baseline.h{}".format(j),
                                          bias=True)
                else:
                    net = tflearn.conv_2d(net, prev_nFilter, 3, strides=strides, reuse=False,
                                          scope="baseline.h{}".format(j), activation='relu', bias=True)
                prev_nFilter = nFilter
            sz = self.config.output_num
            std = 1.0 / np.sqrt(sz)
            probs = tflearn.fully_connected(net, sz, activation='sigmoid', weight_decay=self.config.weight_decay,
                                            weights_init=tfi.variance_scaling(),
                                            bias_init=tfi.zeros(),
                                            regularizer='L2',
                                            reuse=False,
                                            scope=("baseline.fc"))
        return probs

    def cnn_prediction_network(self, xinput=None, reuse=False):

        # input = tf.concat((xinput, yinput), axis=1)
        net = xinput
        j = 0
        with tf.variable_scope("pred"):
            net = tf.reshape(net, shape=(-1, self.config.image_width, self.config.image_height, 1))

            for (nf, fs, st) in self.config.cnn_layer_info:
                net = tflearn.conv_2d(net, nb_filter=nf, filter_size=fs, strides=st,
                                      padding="same", scope=("conv" + str(j)), activation=tf.nn.relu, reuse=reuse)
                # net = tflearn.max_pool_2d(net, kernel_size=[2,2], strides=2)
                # net = tflearn.batch_normalization(net, scope=("bn"+ str(j)), reuse=reuse)
                j = j + 1

            j = 0

            for (sz, a) in self.config.pred_layer_info:
                net = tflearn.fully_connected(net, sz, regularizer='L2',
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              bias_init=tfi.zeros(), reuse=reuse,
                                              scope=("ph." + str(j)))
                net = tf.nn.relu(net)
                net = tflearn.layers.dropout(net, 1 - self.config.dropout)
                j = j + 1

            net = tflearn.fully_connected(net, self.config.output_num * self.config.dimension, activation='linear',
                                          weight_decay=self.config.weight_decay,
                                          weights_init=tfi.variance_scaling(),
                                          bias_init=tfi.zeros(), reuse=reuse,
                                          regularizer='L2',
                                          scope=("ph.fc"))
            if self.config.dimension == 1:
                return tf.nn.sigmoid(net)
            else:
                cat_output = tf.reshape(net, (-1, self.config.output_num, self.config.dimension))
                return tf.nn.softmax(cat_output, dim=2)

    def get_prediction_network(self, xinput=None, reuse=False):
        raise NotImplementedError

    def soft_predict(self, xinput=None, train=False):
        yp = self.sess.run(self.yp, feed_dict={self.x: xinput})
        return yp

    def map_predict(self, xinput=None, train=False):
        tflearn.is_training(train, self.sess)
        yp = self.sess.run(self.yp, feed_dict={self.x: xinput})
        if self.config.loglevel > 1:
            print(yp)
        if self.config.dimension > 1:
            return np.argmax(yp, 2)
        else:
            return yp

    def pred_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def createOptimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)

    def construct(self):

        logits = self.get_prediction_network(self.x)  # horses_network(self.x)
        if self.config.dimension == 1:
            self.objective = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                                   labels=self.y)) + self.config.l2_penalty * self.get_l2_loss()
            self.yp = tf.nn.sigmoid(logits)
        else:
            logits = tf.reshape(logits, (-1, self.config.output_num, self.config.dimension))
            self.objective = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf.reshape(self.y, (-1, self.config.output_num, self.config.dimension)))) \
                             + self.config.l2_penalty * self.get_l2_loss()
            self.yp = tf.nn.softmax(logits, dim=2)

        # self.yp = self.get_prediction_network(self.x)
        # self.objective = self.get_loss(self.y, self.yp) + self.config.l2_penalty * self.get_l2_loss()


        # pred = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="pred")
        # xpred = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="xpred") +  pred

        self.train_step = self.optimizer.minimize(self.objective)
        # self.train_xpred = self.optimizer.minimize(self.objective, var_list=xpred)

        return self

    def set_train_iter(self, iter):
        self.train_iter = iter

    def get_l2_loss(self):
        loss = 0.0
        en_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for v in en_vars:
            loss += tf.nn.l2_loss(v)
        return loss

    def print_vars(self):
        for v in self.pred_variables():
            print(v)

    def train_batch(self, xbatch=None, ybatch=None, verbose=0):
        tflearn.is_training(True, self.sess)
        if self.config.dimension > 1:
            yt_ind = self.var_to_indicator(ybatch)
            yt_ind = np.reshape(yt_ind, (-1, self.config.output_num * self.config.dimension))
        else:
            yt_ind = ybatch  # np.reshape(ybatch, (-1, self.config.output_num, 1))
        feeddic = {self.x: xbatch, self.y: yt_ind,
                   self.labels: ybatch,
                   self.learning_rate_ph: self.config.learning_rate,
                   self.dropout_ph: self.config.dropout}
        # if self.train_iter < self.config.pretrain_iter:
        #  _, o = self.sess.run([self.train_xpred, self.objective], feed_dict=feeddic)
        # else:
        _, o = self.sess.run([self.train_step, self.objective], feed_dict=feeddic)

        if verbose > 0:
            print(self.train_iter, o)
        return o

    def var_to_indicator(self, vd):
        size = np.shape(vd)
        cat = np.zeros((size[0], self.config.output_num, self.config.dimension))
        for i in range(size[0]):
            for j in range(self.config.output_num):
                k = vd[i, j]
                cat[i, j, int(k)] = 1
        return np.reshape(cat, (size[0], self.config.output_num, self.config.dimension))

    def save(self, path):
        self.saver.save(self.sess, path)

    def restore(self, path):
        self.saver.restore(self.sess, path)

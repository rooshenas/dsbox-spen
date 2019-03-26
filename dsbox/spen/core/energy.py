import tensorflow as tf
import tflearn
import tflearn.initializations as tfi
import numpy as np


class EnergyModel:
    def __init__(self, config):
        self.config = config

    def get_feature_net_rnn(self, xinput, reuse=False):
        with tf.variable_scope("bi-lstm") as scope:
            if reuse:
                scope.reuse_variables()

            cell_fw = tf.contrib.rnn.LSTMCell(self.config.lstm_hidden_size, reuse=reuse)
            #cell_bw = tf.contrib.rnn.LSTMCell(self.config.lstm_hidden_size, reuse=reuse)
            #(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            #    cell_fw, cell_bw, xinput, dtype=tf.float32, parallel_iterations=10)
            #output = tf.concat([output_fw, output_bw], axis=-1)

            output,states = tf.nn.dynamic_rnn(cell_fw, xinput, dtype=tf.float32)
            # output = tf.contrib.layers.layer_norm(output, reuse=reuse, scope=("ln0"))
            # output = tf.nn.dropout(output, 1- self.config.dropout)

        #nsteps = tf.shape(output)[1]
        print(output.get_shape().as_list())

        with tf.variable_scope("proj") as scope:
            # if reuse:
            #     scope.reuse_variables()
            # W = tf.get_variable("PW", dtype=tf.float32,
            #                     shape=[2 * self.config.lstm_hidden_size, self.config.dimension])
            #
            # b = tf.get_variable("Pb", shape=[self.config.dimension],
            #                     dtype=tf.float32, initializer=tf.zeros_initializer())
            #
            # output = tf.reshape(output, [-1, 2 * self.config.lstm_hidden_size])
            # pred = tf.matmul(output, W) + b
            # net = tf.reshape(pred, [-1, nsteps * self.config.dimension])
            #
            # print net.get_shape().as_list(), output.get_shape().as_list(), pred.get_shape().as_list()
            j=0
            net = output
            for (sz, a) in self.config.layer_info:
                net = tflearn.fully_connected(net, sz, activation=a,
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              bias_init=tfi.zeros(), regularizer='L2', reuse=reuse,
                                              scope=("fx.h" + str(j)))
                j = j + 1
            logits = tflearn.fully_connected(net, self.config.output_num*self.config.dimension, activation='relu', regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(), bias_init=tfi.zeros(),
                                             reuse=reuse, scope="fx.fc")

        return logits

    def get_feature_net_mlp(self, xinput, output_num, embedding=None, reuse=False):
        # with tf.variable_scope("spen/fx") as scope:

        net = xinput
        j = 0
        for (sz, a) in self.config.layer_info:
            net = tflearn.fully_connected(net, sz,
                                          weight_decay=self.config.weight_decay, activation=a,
                                          weights_init=tfi.variance_scaling(),
                                          bias_init=tfi.zeros(), regularizer='L2', reuse=reuse, scope=("fx.h" + str(j)))
            # net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope=("bn.f" + str(j)))
            #net = tflearn.activations.relu(net)
            net = tflearn.dropout(net, 1.0 - self.config.dropout)
            j = j + 1
        logits = tflearn.fully_connected(net, output_num, activation='linear', regularizer='L2',
                                          weight_decay=self.config.weight_decay,
                                          weights_init=tfi.variance_scaling(), bias_init=tfi.zeros(),
                                          reuse=reuse, scope="fx.fc")
        #logits = net
        return logits

    def get_init_mlp(self, xinput, output_num, embedding=None, reuse=False):
        with tf.variable_scope("spen/init") as scope:
            net = xinput
            j = 0
            for (sz, a) in self.config.layer_info:
                net = tflearn.fully_connected(net, sz,
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              bias_init=tfi.zeros(), regularizer='L2', reuse=reuse,
                                              scope=("fx.h" + str(j)))
                # net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope=("bn.f" + str(j)))
                net = tflearn.activations.relu(net)

                j = j + 1
            features = net
            net = tflearn.dropout(net, 1.0 - self.config.dropout)
            logits = tflearn.fully_connected(net, output_num, activation='relu', regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(), bias_init=tfi.zeros(),
                                             reuse=reuse, scope=("fx.h" + str(j)))
        return logits, features

    def get_global_energy(self, xinput=None, yinput=None, embedding=None, reuse=False):
        if embedding is not None:
            xinput = tf.cast(xinput, tf.int32)
            xinput = tf.nn.embedding_lookup(embedding, xinput)

        with tf.variable_scope(self.config.spen_variable_scope) as scope:
            logits = tflearn.fully_connected(xinput, yinput.get_shape().as_list()[-1], activation='relu',
                                             regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(), bias_init=tfi.zeros(),
                                             reuse=reuse, scope="fx.fc")

            mult_ = tf.multiply(logits, yinput)
            local_e = tflearn.fully_connected(mult_, 1, activation='linear', regularizer='L2',
                                              weight_decay=self.config.weight_decay,
                                              bias=False,
                                              weights_init=tfi.variance_scaling(),
                                              reuse=reuse, scope=("en.l"))
        # local_e = self.config.inf_penalty * tf.reduce_sum(tf.square(yinput-xinput),1)
        # local_e = self.config.inf_penalty * tf.reduce_sum(tf.square(yinput))
        with tf.variable_scope(self.config.spen_variable_scope) as scope:
            j = 0
            # net = tf.concat((yinput, xinput), axis=1)
            net = yinput
            for (sz, a) in self.config.en_layer_info:
                net = tflearn.fully_connected(net, sz,
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              bias_init=tfi.zeros(), reuse=reuse, regularizer='L2',
                                              scope=("en.h" + str(j)))
                net = tf.log(tf.exp(net) + self.config.temperature)
                j = j + 1
                # net = tf.contrib.layers.layer_norm(net, reuse=reuse, scope=("ln"+str(j)))
            global_e = tflearn.fully_connected(net, 1, activation='linear', weight_decay=self.config.weight_decay,
                                               weights_init=tfi.zeros(), bias=False,
                                               reuse=reuse, regularizer='L2',
                                               scope=("en.g"))
            return tf.squeeze(global_e + local_e)

    def get_energy_joint_mlp(self, xinput=None, yinput=None, embedding=None, reuse=False):
        net = tf.concat((yinput, xinput), axis=1)
        with tf.variable_scope(self.config.spen_variable_scope) as scope:
            j = 0
            for (sz, a) in self.config.en_layer_info:
                net = tflearn.fully_connected(net, sz,
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              activation=a,
                                              bias=False,
                                              reuse=reuse, regularizer='L2',
                                              scope=("en.h" + str(j)))

                # net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope=("bn.d" + str(j)))
                # net = tflearn.activations.softplus(net)
                # net = tf.log(tf.exp(net) + self.config.temperature)
                # net = tflearn.dropout(net, 1.0 - self.config.dropout)
                # net = tf.contrib.layers.layer_norm(net, reuse=reuse, scope=("ln"+str(j)))
                j = j + 1
            global_e = tf.squeeze(
                tflearn.fully_connected(net, 1, activation='linear', weight_decay=self.config.weight_decay,
                                        weights_init=tfi.variance_scaling(), bias=False,
                                        reuse=reuse, regularizer='L2',
                                        scope=("en.g")))
            return tf.squeeze(global_e)

    def get_energy_mlp(self, xinput=None, yinput=None, embedding=None, reuse=False):
        output_size = yinput.get_shape().as_list()[-1]
        with tf.variable_scope(self.config.spen_variable_scope):
            with tf.variable_scope(self.config.fx_variable_scope) as scope:
                logits = self.get_feature_net_mlp(xinput, output_size, reuse=reuse)

                mult = logits * yinput
                local_e = tf.reduce_sum(mult, axis=1)
                #print "here", logits.get_shape().as_list(), yinput.get_shape().as_list(), mult.get_shape().as_list(), local_e.get_shape().as_list()

                #local_e = tflearn.fully_connected(mult, 1, activation='linear', regularizer='L2',
                #                                  weight_decay=self.config.weight_decay,
                #                                  weights_init=tfi.variance_scaling(),
                #                                  bias=False,
                #                                  bias_init=tfi.zeros(), reuse=reuse, scope=("en.l"))
            with tf.variable_scope(self.config.en_variable_scope) as scope:
                j = 0
                net = yinput
                for (sz, a) in self.config.en_layer_info:
                    net = tflearn.fully_connected(net, sz,
                                                  weight_decay=self.config.weight_decay,
                                                  weights_init=tfi.variance_scaling(),
                                                  activation=a,
                                                  bias=False,
                                                  reuse=reuse, regularizer='L2',
                                                  scope=("en.h" + str(j)))

                    # net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope=("bn.d" + str(j)))
                    # net = tflearn.activations.softplus(net)
                    #net = tf.log(tf.exp(net) + self.config.temperature)
                    # net = tflearn.dropout(net, 1.0 - self.config.dropout)
                    # net = tf.contrib.layers.layer_norm(net, reuse=reuse, scope=("ln"+str(j)))
                    j = j + 1
                global_e = tf.squeeze(tflearn.fully_connected(net, 1, activation='linear', weight_decay=self.config.weight_decay,
                                                   weights_init=tfi.variance_scaling(), bias=False,
                                                   reuse=reuse, regularizer='L2',
                                                   scope=("en.g")))

        return tf.squeeze(tf.add(local_e, global_e))

    def get_energy_mlp_local(self, xinput=None, yinput=None, embedding=None, reuse=False):
        output_size = yinput.get_shape().as_list()[-1]
        with tf.variable_scope(self.config.spen_variable_scope):
            with tf.variable_scope(self.config.fx_variable_scope) as scope:
                logits = self.get_feature_net_mlp(xinput, output_size, reuse=reuse)
                mult = tf.multiply(logits, yinput)
                # local_e = tf.reduce_sum(mult, axis=1)
                j = 0
                net = mult
                for (sz, a) in self.config.en_layer_info:
                        net = tflearn.fully_connected(net, sz,
                                                      weight_decay=self.config.weight_decay,
                                                      weights_init=tfi.variance_scaling(),
                                                      activation=a,
                                                      bias=False,
                                                      reuse=reuse, regularizer='L2',
                                                      scope=("en.h" + str(j)))

                        # net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope=("bn.d" + str(j)))
                        # net = tflearn.activations.softplus(net)
                        # net = tf.log(tf.exp(net) + self.config.temperature)
                        # net = tflearn.dropout(net, 1.0 - self.config.dropout)
                        # net = tf.contrib.layers.layer_norm(net, reuse=reuse, scope=("ln"+str(j)))
                        j = j + 1

                global_e = tflearn.fully_connected(net, 1, activation='linear',
                                                       weight_decay=self.config.weight_decay,
                                                       weights_init=tfi.variance_scaling(), bias=False,
                                                       reuse=reuse, regularizer='L2',
                                                       scope=("en.g"))


        return tf.squeeze(global_e)

    def get_energy_mlp_emb(self, xinput, yinput, embedding=None, reuse=False):
        xinput = tf.cast(xinput, tf.int32)
        xinput = tf.nn.embedding_lookup(embedding, xinput)
        return self.get_energy_mlp(xinput=xinput, yinput=yinput, reuse=reuse)

    def get_feature_net_mlp_emb(self, xinput, output_num, embedding=None, reuse=False):
        xinput = tf.cast(xinput, tf.int32)
        xinput = tf.nn.embedding_lookup(embedding, xinput)
        return self.get_feature_net_mlp(xinput, output_num, reuse)

    def get_energy_rnn_emb(self, xinput, yinput, embedding=None, reuse=False):
        xinput = tf.cast(xinput, tf.int32)
        xinput = tf.nn.embedding_lookup(embedding, xinput)
        with tf.variable_scope(self.config.spen_variable_scope):
            with tf.variable_scope(self.config.fx_variable_scope) as scope:
                logits = self.get_feature_net_rnn(xinput, reuse=reuse)
                mult = tf.multiply(logits, yinput)


            with tf.variable_scope(self.config.en_variable_scope) as scope:

                local_e = tflearn.fully_connected(mult, 1, activation='linear', weight_decay=self.config.weight_decay,
                                                  weights_init=tfi.variance_scaling(),
                                                  bias=None,
                                                  bias_init=tfi.zeros(), reuse=reuse, scope=("en.l"))

                net = yinput
                j = 0

                for (sz, a) in self.config.en_layer_info:
                  net = tflearn.fully_connected(net, sz,activation=a,
                                                weight_decay=self.config.weight_decay,
                                                weights_init=tfi.variance_scaling(),
                                                bias_init=tfi.zeros(), reuse=reuse,
                                                scope=("en.h" + str(j)))
                  j = j + 1


                global_e = tflearn.fully_connected(net, 1, activation='linear', weight_decay=self.config.weight_decay,
                                                   weights_init=tfi.zeros(),
                                                   bias=None, reuse=reuse,
                                                   scope="en.g")


        net = local_e + global_e
        return tf.squeeze(net)

    def get_energy_rnn_joint_emb(self, xinput, yinput, embedding=None, reuse=False):
        xinput = tf.cast(xinput, tf.int32)
        xinput = tf.nn.embedding_lookup(embedding, xinput)
        with tf.variable_scope(self.config.spen_variable_scope):
            with tf.variable_scope(self.config.fx_variable_scope) as scope:
                logits = self.get_feature_net_rnn(xinput, reuse=reuse)
                mult = tf.multiply(logits, yinput)

            with tf.variable_scope(self.config.en_variable_scope) as scope:

                # local_e = tflearn.fully_connected(mult, 1, activation='linear', weight_decay=self.config.weight_decay,
                #                                   weights_init=tfi.variance_scaling(),
                #                                   bias=None,
                #                                   bias_init=tfi.zeros(), reuse=reuse, scope=("en.l"))
                #
                net = mult
                j = 0

                for (sz, a) in self.config.en_layer_info:
                    net = tflearn.fully_connected(net, sz, activation=a,
                                                  weight_decay=self.config.weight_decay,
                                                  weights_init=tfi.variance_scaling(),
                                                  bias_init=tfi.zeros(), reuse=reuse,
                                                  scope=("en.h" + str(j)))
                    j = j + 1

                global_e = tflearn.fully_connected(net, 1, activation='linear', weight_decay=self.config.weight_decay,
                                                   weights_init=tfi.zeros(),
                                                   bias=None, reuse=reuse,
                                                   scope="en.g")

        net = global_e
        return tf.squeeze(net)

    def get_enerey_rnn_mlp_emb(self, xinput, yinput, embedding=None, reuse=False):
        xinput.cast(xinput, tf.int32)
        xinput.nn.embedding_lookup(embedding, xinput)
        output_size = yinput.get_shape().as_list()[-1]
        with tf.variable_scope(self.config.spen_variable_scope):
            with tf.variable_scope(self.config.fx_variable_scope) as scope:
                logits2 = self.get_feature_net_rnn(xinput, reuse=reuse)
                logits3 = self.get_feature_net_mlp(xinput, output_size, reuse=reuse)  # + logits2

                # mult_ = tf.multiply(logits, yinput)
                mult2 = tf.multiply(logits2, yinput)
                mult3 = tf.multiply(logits3, yinput)

                local_e3 = tflearn.fully_connected(mult3, 1, activation='linear', weight_decay=self.config.weight_decay,
                                                   weights_init=tfi.variance_scaling(),
                                                   bias=None,
                                                   bias_init=tfi.zeros(), reuse=reuse, scope=("fx3.b0"))

                local_e2 = tflearn.fully_connected(mult2, 1, activation='linear', weight_decay=self.config.weight_decay,
                                                   weights_init=tfi.variance_scaling(),
                                                   bias=None,
                                                   bias_init=tfi.zeros(), reuse=reuse, scope=("fx2.b0"))


                # local_e = tflearn.fully_connected(mult_, 1, activation='linear', weight_decay=self.config.weight_decay,
                #                                  weights_init=tfi.variance_scaling(),
                #                                  bias=None,
                #                                  bias_init=tfi.zeros(), reuse=reuse, scope=("fx.b0"))

            j = 0

            with tf.variable_scope(self.config.en_variable_scope) as scope:
                net = yinput
                j = 0

                for (sz, a) in self.config.en_layer_info:
                    # std = np.sqrt(2.0) / np.sqrt(sz)
                    net = tflearn.fully_connected(net, sz, activation=a,
                                                  weight_decay=self.config.weight_decay,
                                                  weights_init=tfi.variance_scaling(),
                                                  bias_init=tfi.zeros(), reuse=reuse,
                                                  scope=("en.h" + str(j)))
                    #  net = tflearn.activations.relu(net)
                    # net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope=("bn.en" + str(j)))

                    j = j + 1

                global_e = tflearn.fully_connected(net, 1, activation='linear', weight_decay=self.config.weight_decay,
                                                   weights_init=tfi.zeros(),
                                                   bias_init=tfi.zeros(), reuse=reuse,
                                                   scope=("en.h" + str(j)))
                # en = tflearn.layers.normalization.batch_normalization(en, reuse=ru, scope=("en." + str(j)))


                if reuse:
                    scope.reuse_variables()

                net = global_e + local_e3 + local_e2  # + local_e + local_e3

            return tf.squeeze(net)

    def get_feature_net_cnn(self, input, output_num, reuse=False):
        net = input
        netw = tf.nn.embedding_lookup(self.W, net)
        net_expanded = tf.expand_dims(netw, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope("max-pooling") as scope:
                if reuse:
                    scope.reuse_variables()
                filter_shape = [filter_size, self.embedding_size, 1, self.config.num_filters]
                W = tf.get_variable(initializer=tf.truncated_normal(filter_shape, stddev=0.1),
                                    name=("W" + ("-conv-maxpool-%s" % filter_size)))
                b = tf.get_variable(initializer=tf.constant(0.1, shape=[self.config.num_filters]),
                                    name=("b" + ("-conv-maxpool-%s" % filter_size)))
                conv = tf.nn.conv2d(net_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name=("conv" + ("-conv-maxpool-%s" % filter_size)))
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        net = tf.reshape(h_pool, [-1, num_filters_total])
        net = tflearn.dropout(net, 1.0 - self.config.dropout)
        j = 0
        net2 = netw
        sz = self.config.output_num

        logits = tflearn.fully_connected(net, sz, activation='linear', weight_decay=self.config.weight_decay,
                                         weights_init=tfi.variance_scaling(),
                                         bias_init=tfi.zeros(), reuse=reuse, scope=("fx.h-cnn" + str(j)))
        return logits

    def softmax_prediction_network(self, input=None, xinput=None, reuse=False):
        net = input
        j = 0
        with tf.variable_scope("pred") as scope:
            for (sz, a) in self.config.pred_layer_info:
                net = tflearn.fully_connected(net, sz, regularizer='L2', activation=a,
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              bias_init=tfi.zeros(), reuse=reuse,
                                              scope=("ph." + str(j)))
                # net = tf.nn.relu(net)
                net = tflearn.layers.dropout(net, 1 - self.config.dropout)
                j = j + 1

            net = tflearn.fully_connected(net, self.config.output_num * self.config.dimension, activation='linear',
                                          weight_decay=self.config.weight_decay,
                                          weights_init=tfi.variance_scaling(),
                                          reuse=reuse,
                                          bias=None,
                                          regularizer='L2',
                                          scope=("ph.fc"))

        if self.config.dimension == 1:
            return tf.nn.sigmoid(net)
        else:
            cat_output = tf.reshape(net, (-1, self.config.output_num, self.config.dimension))
            return tf.nn.softmax(cat_output, dim=2)

    def sigmoid_prediction_network(self, input=None, xinput=None, reuse=False):
        net = input
        net = tflearn.layers.dropout(net, 0.8)
        j = 0
        with tf.variable_scope("pred") as scope:
            for (sz, a) in self.config.pred_layer_info:
                net = tflearn.fully_connected(net, sz, regularizer='L2', activation=a,
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              bias_init=tfi.zeros(), reuse=reuse,
                                              scope=("ph." + str(j)))

                net = tflearn.layers.dropout(net, 1 - self.config.dropout)
                j = j + 1

            net = tflearn.fully_connected(net, self.config.output_num * self.config.dimension, activation='linear',
                                          weight_decay=self.config.weight_decay,
                                          weights_init=tfi.variance_scaling(),
                                          bias=False,
                                          reuse=reuse,
                                          regularizer='L2',
                                          scope=("ph.fc"))

        return tf.nn.sigmoid(net)

    def linear_prediction_network(self, input=None, xinput=None, reuse=False):
        net = input
        # net = tflearn.layers.dropout(net, 0.8)
        j = 0
        with tf.variable_scope("pred") as scope:
            for (sz, a) in self.config.pred_layer_info:
                net = tflearn.fully_connected(net, sz, regularizer='L2',
                                              activation=a,
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              bias_init=tfi.zeros(), reuse=reuse,
                                              scope=("ph." + str(j)))
                net = tflearn.layers.dropout(net, 1 - self.config.dropout)
                j = j + 1

            net = tflearn.fully_connected(net, self.config.output_num * self.config.dimension, activation='linear',
                                          weight_decay=self.config.weight_decay,
                                          weights_init=tfi.variance_scaling(),
                                          bias=False,
                                          reuse=reuse,
                                          regularizer='L2',
                                          scope=("ph.fc"))

        return net

    def sigmoid2_prediction_network(self, input=None, xinput=None, reuse=False):
        net = input
        j = 0
        with tf.variable_scope("pred") as scope:
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

        return tf.nn.sigmoid(net)

    def joint_prediction_network(self, input=None, xinput=None, reuse=False):
        net = tf.concat((input, xinput), axis=1)
        j = 0
        with tf.variable_scope("pred") as scope:
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
            # return tf.nn.softmax(cat_output, dim=2), cat_output

    def softmax_network(self, hidden_vars, reuse=False):
        net = hidden_vars
        cat_output = tf.reshape(net, (-1, self.config.output_num, self.config.dimension))
        return tf.nn.softmax(cat_output, dim=2)

    def energy_cnn_image(self, xinput=None, yinput=None, embedding=None, reuse=False):
        image_size = tf.cast(tf.sqrt(tf.cast(tf.shape(yinput)[1], tf.float64)), tf.int32)
        output_size = yinput.get_shape().as_list()[-1]
        with tf.variable_scope(self.config.spen_variable_scope):
            yinput = tf.reshape(yinput, shape=(-1, image_size, image_size))
            conv1 = tf.layers.conv2d(
                inputs=tf.expand_dims(yinput, axis=3),
                filters=8,
                kernel_size=[3, 3],
                padding="same",
                name="conv1",
                activation=tf.nn.relu, reuse=reuse)
            conv1 = tf.nn.dropout(conv1, 1.0 - self.config.dropout)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="spen/pool1")
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=16,
                kernel_size=[3, 3],
                padding="same",
                name="conv2",
                activation=tf.nn.relu, reuse=reuse)
            conv2 = tf.nn.dropout(conv2, 1.0 - self.config.dropout)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="spen/pool2")

            conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=32,
                kernel_size=[3, 3],
                padding="same",
                name="conv3",
                activation=tf.nn.relu, reuse=reuse)
            conv3 = tf.nn.dropout(conv3, 1.0 - self.config.dropout)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, name="spen/pool3")
            self.state_dim = 128

            self.encode_embeddings = tf.reshape(pool3, [100, self.state_dim])

            energy = tflearn.fully_connected(self.encode_embeddings, 1, activation='linear', regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(), bias=False,
                                             reuse=reuse, scope=("fc"))

            # with tf.variable_scope(self.config.fx_variable_scope):
            #  logits = self.get_feature_net_mlp(pool3, output_size, reuse=reuse)

            # local_e = -tf.reduce_sum(tf.square(yinput - logits),1)

            # mult = tf.multiply(logits, yinput)

            # local_e = tflearn.fully_connected(, 1, activation='linear', weight_decay=self.config.weight_decay,
            #                                   weights_init=tfi.variance_scaling(),
            #                                   bias=None,
            #                                   bias_init=tfi.zeros(), reuse=reuse, scope="en.l")

            # with tf.variable_scope(self.config.en_variable_scope) as scope:
            #  net = yinput
            #  j = 0

            #  for (sz, a) in self.config.en_layer_info:
            #    net = tflearn.fully_connected(net, sz, activation=a,
            #                                  weight_decay=self.config.weight_decay,
            #                                  weights_init=tfi.variance_scaling(),
            #                                  bias_init=tfi.zeros(), reuse=reuse,
            #                                  scope=("en.h" + str(j)))
            #    j = j + 1

            #  global_e = tflearn.fully_connected(net, 1, activation='linear', weight_decay=self.config.weight_decay,
            #                                     weights_init=tfi.zeros(),
            #                                     bias_init=tfi.zeros(), reuse=reuse,
            #                                     scope=("en.h" + str(j)))

        # net = local_e + global_e

        return tf.squeeze(energy)

    def cnn_prediction_network(self, input=None, xinput=None, embedding=None, reuse=False):

        # net = tf.concat((input, xinput), axis=1)
        net = xinput
        j = 0
        with tf.variable_scope("pred"):
            net = tf.reshape(net, shape=(-1, self.config.image_width, self.config.image_height, 1))

            for (nf, fs, st) in self.config.cnn_layer_info:
                net = tflearn.conv_2d(net, nb_filter=nf, filter_size=fs, strides=st,
                                      padding="same", scope=("conv" + str(j)), activation=tf.nn.relu, reuse=reuse)
                # net = tflearn.batch_normalization(net, scope=("bn"+ str(j)), reuse=reuse)
                j = j + 1

            net = tflearn.fully_connected(net, self.config.hidden_num, activation='relu', regularizer='L2',
                                          weight_decay=self.config.weight_decay,
                                          weights_init=tfi.variance_scaling(), bias=False,
                                          reuse=reuse, scope=("fc.h"))
            net = tf.concat((net, input), axis=1)

            # net = tf.multiply(net, input)
            # with tf.variable_scope(self.config.fx_variable_scope):
            #  logits = self.get_feature_net_mlp(pool3, output_size, reuse=reuse)

            # local_e = -tf.reduce_sum(tf.square(yinput - logits),1)
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

    def get_init_cnn(self, xinput, output_size, embedding=None, reuse=False):
        net = xinput
        j = 0
        with tf.variable_scope("spen/init"):
            net = tf.reshape(net, shape=(-1, self.config.image_width, self.config.image_height, 3))

            for (nf, fs, st) in self.config.cnn_layer_info:
                net = tflearn.conv_2d(net, nb_filter=nf, filter_size=fs, strides=st,
                                      padding="same", scope=("conv" + str(j)), activation=tf.nn.relu, reuse=reuse)

                j = j + 1
            net = tflearn.layers.dropout(net, 1 - self.config.dropout)
            j = 0
            for (sz, a) in self.config.layer_info:
                net = tflearn.fully_connected(net, sz,
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              activation=a,
                                              bias_init=tfi.zeros(), regularizer='L2', reuse=reuse,
                                              scope=("fx.h" + str(j)))
                j = j + 1

            logits = tflearn.fully_connected(net, output_size, activation='linear', regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(), bias=False,
                                             reuse=reuse, scope=("fx.fc"))
        return logits

    def get_init_horses(self, xinput, output_size, embedding=None, reuse=False):
        net = xinput
        j = 0
        with tf.variable_scope("spen/init"):
            net = tf.reshape(net, shape=(-1, self.config.image_width, self.config.image_height, 3))

            net = tf.layers.conv2d(inputs=net, filters=8,
                                   kernel_size=[5, 5], padding="same", name="conv1",
                                   activation=tf.nn.relu, reuse=reuse)
            #  net = tf.layers.batch_normalization()
            net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool1")
            net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            net = tf.layers.conv2d(inputs=net, filters=16,
                                   kernel_size=[5, 5], padding="same", name="conv2",
                                   activation=tf.nn.relu, reuse=reuse)

            net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool2")
            net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            # net = tf.layers.conv2d(inputs=net, filters=32,
            #                       kernel_size=[5, 5], padding="same", name="conv3",
            #                       activation=tf.nn.relu, reuse=reuse)

            # net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            net = tf.reshape(net, [-1, (self.config.image_width / 4) * (self.config.image_height / 4) * 16])

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

            logits = tflearn.fully_connected(net, output_size, activation='linear', regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(), bias=False,
                                             reuse=reuse, scope=("fx.fc"))
        return logits, net

    def get_energy_cnn(self, xinput=None, yinput=None, embedding=None, reuse=False):
        print(("tet", xinput.get_shape().as_list()))
        print(("ytet", yinput.get_shape().as_list()))
        # input = tf.concat((xinput, yinput), axis=1)
        input = xinput
        print(("as", input.get_shape().as_list()))
        # image_size = tf.cast(tf.sqrt(tf.cast(tf.shape(input)[1], tf.float64)), tf.int32)
        output_size = yinput.get_shape().as_list()[-1]
        batch_size = xinput.get_shape().as_list()[0]
        print(("batch size", batch_size))
        net = input
        j = 0
        with tf.variable_scope(self.config.spen_variable_scope):
            net = tf.reshape(net, shape=(-1, self.config.image_width, self.config.image_height, 1))

            for (nf, fs, st) in self.config.cnn_layer_info:
                net = tflearn.conv_2d(net, nb_filter=nf, filter_size=fs, strides=st,
                                      padding="same", scope=("conv" + str(j)), activation=tf.nn.relu, reuse=reuse)
                # net = tflearn.batch_normalization(net)
                j = j + 1

            j = 0
            for (sz, a) in self.config.layer_info:
                net = tflearn.fully_connected(net, sz,
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              activation=a,
                                              bias_init=tfi.zeros(), regularizer='L2', reuse=reuse,
                                              scope=("fx.h" + str(j)))
                # net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope=("bn.f" + str(j)))
                # net = tflearn.activations.relu(net)
                # net = tflearn.dropout(net, 1.0 - self.config.dropout)
                j = j + 1
            # conv1 = tf.nn.dropout(conv1, 1.0 - self.config.dropout)
            # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="spen/pool1")
            # conv2 = tf.layers.conv2d(
            #   inputs=pool1,
            #   filters=64,
            #   kernel_size=[5, 5],
            #   padding="same",
            #   name="conv2",
            #   activation=tf.nn.relu, reuse=reuse)
            # conv2 = tf.nn.dropout(conv2, 1.0 -self.config.dropout)
            # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="spen/pool2")

            # conv3 = tf.layers.conv2d(
            #   inputs=pool2,
            #   filters=32,
            #   kernel_size=[3, 3],
            #   padding="valid",
            #   name="conv3",
            #   activation=tf.nn.relu, reuse=reuse)
            # conv3 = tf.nn.dropout(conv3, 1.0 - self.config.dropout)
            # pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, name="spen/pool3")
            # self.state_dim = 64*(self.config.image_height/4)*(self.config.image_width/4)
            # print (pool3.get_shape().as_list())
            # self.encode_embeddings = tf.reshape(pool2, shape=(-1, self.state_dim))




            logits = tflearn.fully_connected(net, output_size, activation='linear', regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(), bias=False,
                                             reuse=reuse, scope=("fx.fc"))

            # with tf.variable_scope(self.config.fx_variable_scope):
            #  logits = self.get_feature_net_mlp(pool3, output_size, reuse=reuse)

            # local_e = -tf.reduce_sum(tf.square(yinput - logits),1)

            mult = tf.multiply(logits, yinput)

            local_e = tflearn.fully_connected(mult, 1, activation='linear', weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              bias=None,
                                              bias_init=tfi.zeros(), reuse=reuse, scope="en.l")

            with tf.variable_scope(self.config.en_variable_scope) as scope:
                net = yinput
                j = 0

                for (sz, a) in self.config.en_layer_info:
                    net = tflearn.fully_connected(net, sz, activation=a,
                                                  weight_decay=self.config.weight_decay,
                                                  weights_init=tfi.variance_scaling(),
                                                  bias_init=tfi.zeros(), reuse=reuse,
                                                  scope=("en.h" + str(j)))
                    j = j + 1

                global_e = tflearn.fully_connected(net, 1, activation='linear', weight_decay=self.config.weight_decay,
                                                   weights_init=tfi.zeros(),
                                                   bias_init=tfi.zeros(), reuse=reuse,
                                                   scope=("en.h" + str(j)))

        energy = local_e + global_e

        return tf.squeeze(energy)



    def get_energy_horses(self, xinput=None, yinput=None, embedding=None, reuse=False):
        j = 0
        net = xinput
        print(("yinput:", yinput.get_shape().as_list()))
        with tf.variable_scope("spen/fx"):
            net = tf.reshape(net, shape=(-1, self.config.image_width, self.config.image_height, 3))
            mask = tf.reshape(yinput, shape=(-1, self.config.image_width, self.config.image_width, 2))
            #mask = tf.expand_dims(mask[:, :, :, 1],-1)
            print(("mask:", mask.get_shape().as_list()))
            net = tf.concat((net, mask), axis=3)
            print((net.get_shape().as_list()))

            net = tf.layers.conv2d(inputs=net, filters=64,
                                   kernel_size=[5, 5], strides=1, padding="same", name="conv00",
                                   activation=tf.nn.relu, reuse=reuse)

            net = tf.layers.conv2d(inputs=net, filters=128,
                                   kernel_size=[5, 5], strides=2, padding="same", name="conv01",
                                   activation=tf.nn.relu, reuse=reuse)

            #net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool1")

            net = tf.layers.conv2d(inputs=net, filters=128,
                                   kernel_size=[5, 5], strides=2, padding="same", name="conv10",
                                   activation=tf.nn.relu, reuse=reuse)

            # net = tf.layers.conv2d(inputs=net, filters=32, strides=1,
            #                        kernel_size=[3, 3], padding="same", name="conv11",
            #                        activation=tf.nn.relu, reuse=reuse)
            #
            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool2")


            #net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool1")
            #  net = tf.layers.batch_normalization()
            #net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool1")
            #net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            # net = tf.layers.conv2d(inputs=net, filters=64*4,
            #                        kernel_size=[3, 3], padding="same", name="conv20", strides=1,
            #                        activation=tf.nn.relu, reuse=reuse)
            #
            # #net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool2")
            # #net = tflearn.local_response_normalization(net)
            # # net = tflearn.dropout(net, 0.8)
            #
            # net = tf.layers.conv2d(inputs=net, filters=64*4,
            #                        kernel_size=[3, 3], padding="same", name="conv21", strides=1,
            #                        activation=tf.nn.relu, reuse=reuse)
            #
            #
            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool3")
            #net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool3")
            #net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            #net = tf.reshape(net, [-1, (self.config.image_width / 8) * (self.config.image_height / 8) * 128])

            net = tf.layers.flatten(net)
            #net = tf.concat((net,yinput),-1)

            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool3")
            #net = tflearn.dropout(net, 1.0 - self.config.dropout)
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

            output = tflearn.fully_connected(net, 1, activation='linear',
                                             regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(), bias=False,
                                             reuse=reuse, scope=("fx.fc"))
        return tf.squeeze(output)

    def get_energy_horses_high(self, xinput=None, yinput=None, embedding=None, reuse=False):
        j = 0
        net = xinput
        print(("yinput:", yinput.get_shape().as_list()))
        with tf.variable_scope("spen/fx"):
            net = tf.reshape(net, shape=(-1, self.config.image_width, self.config.image_height, 3))
            mask = tf.reshape(yinput, shape=(-1, self.config.image_width, self.config.image_width, 2))
            mask = tf.expand_dims(mask[:, :, :, 1], -1)
            print(("mask:", mask.get_shape().as_list()))
            net = tf.concat((net, mask), axis=3)
            print((net.get_shape().as_list()))

            net = tf.layers.conv2d(inputs=net, filters=64,
                                   kernel_size=[3, 3], strides=1, padding="VALID", name="conv0p",
                                   activation=tf.nn.relu, reuse=reuse)

            net = tf.layers.conv2d(inputs=net, filters=64,
                                   kernel_size=[3, 3], strides=2, padding="VALID", name="conv0",
                                   activation=tf.nn.relu, reuse=reuse)

            net = tf.layers.conv2d(inputs=net, filters=128, strides=1,
                                   kernel_size=[3, 3], padding="VALID", name="conv1",
                                   activation=tf.nn.relu, reuse=reuse)



            #  net = tf.layers.batch_normalization()
            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool1")
            # net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            net = tf.layers.conv2d(inputs=net, filters=128,
                                   kernel_size=[3, 3], padding="VALID", name="conv2", strides=2,
                                   activation=tf.nn.relu, reuse=reuse)

            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool2")
            # net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            net = tf.layers.conv2d(inputs=net, filters=256,
                                   kernel_size=[3, 3], padding="VALID", name="conv3", strides=1,
                                   activation=tf.nn.relu, reuse=reuse)

            net = tf.layers.conv2d(inputs=net, filters=256,
                                   kernel_size=[3, 3], padding="VALID", name="conv4", strides=2,
                                   activation=tf.nn.relu, reuse=reuse)
            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool3")
            # net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            # net = tf.reshape(net, [-1, (self.config.image_width / 8) * (self.config.image_height / 8) * 128])

            net = tf.layers.flatten(net)
            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool3")
            # net = tflearn.dropout(net, 1.0 - self.config.dropout)
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

            output = tflearn.fully_connected(net, 1, activation='linear',
                                             regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(), bias=False,
                                             reuse=reuse, scope=("fx.fc"))
        return tf.squeeze(output)

    def get_energy_horses_local(self, xinput=None, yinput=None, embedding=None, reuse=False):
        j = 0
        net = xinput
        with tf.variable_scope("spen/fx"):
            net = tf.reshape(net, shape=(-1, self.config.image_width, self.config.image_height, 3))

            net = tf.layers.conv2d(inputs=net, filters=64,
                                   kernel_size=[5, 5], strides=1, padding="same", name="conv0",
                                   activation=tf.nn.relu, reuse=reuse)

            net = tf.layers.conv2d(inputs=net, filters=64, strides=2,
                                   kernel_size=[5, 5], padding="same", name="conv1",
                                   activation=tf.nn.relu, reuse=reuse)
            #  net = tf.layers.batch_normalization()
            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool1")
            # net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            net = tf.layers.conv2d(inputs=net, filters=128,
                                   kernel_size=[5, 5], padding="same", name="conv2", strides=2,
                                   activation=tf.nn.relu, reuse=reuse)

            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool2")
            # net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            net = tf.layers.conv2d(inputs=net, filters=128,
                                   kernel_size=[5, 5], padding="same", name="conv3", strides=2,
                                   activation=tf.nn.relu, reuse=reuse)
            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool3")
            # net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            # net = tf.reshape(net, [-1, (self.config.image_width / 8) * (self.config.image_height / 8) * 128])

            net = tf.layers.flatten(net)
            #tf.reshape(net, [-1, (self.config.image_width / 8) * (self.config.image_height / 8) * 128])

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

            logits = tflearn.fully_connected(net, yinput.get_shape().as_list()[-1], activation='linear',
                                             regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(), bias=False,
                                             reuse=reuse, scope=("fx.fc"))
            #logits = tf.reshape(logits, (-1, self.config.output_num, self.config.dimension))
            #logits = tf.nn.softmax(logits)
            #unipotential = tf.reshape(-tf.log(logits), (-1, self.config.output_num*self.config.dimension))

        with tf.variable_scope("spen/en") as scope:
            #yinput = tf.nn.softmax(tf.reshape(yinput, (-1, self.config.output_num, self.config.dimension)))
            #yinput = tf.reshape(yinput, (-1, self.config.output_num * self.config.dimension))

            mult = tf.multiply(logits, yinput)
            #local_e = tf.reduce_sum(mult, axis=1)
            local_e = tflearn.fully_connected(mult, 1, activation='linear', regularizer='L2',
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              bias=False,
                                              bias_init=tfi.zeros(), reuse=reuse, scope=("en.l"))

        return tf.squeeze(local_e)


    def get_energy_horses_sep(self, xinput=None, yinput=None, embedding=None, reuse=False):
        j = 0
        net = xinput
        with tf.variable_scope("spen/fx"):
            net = tf.reshape(net, shape=(-1, self.config.image_width, self.config.image_height, 3))

            net = tf.layers.conv2d(inputs=net, filters=64,
                                   kernel_size=[5, 5], strides=1, padding="same", name="conv0",
                                   activation=tf.nn.relu, reuse=reuse)

            net = tf.layers.conv2d(inputs=net, filters=64, strides=2,
                                   kernel_size=[5, 5], padding="same", name="conv1",
                                   activation=tf.nn.relu, reuse=reuse)
            #  net = tf.layers.batch_normalization()
            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool1")
            # net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            net = tf.layers.conv2d(inputs=net, filters=128,
                                   kernel_size=[5, 5], padding="same", name="conv2", strides=2,
                                   activation=tf.nn.relu, reuse=reuse)

            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool2")
            # net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            net = tf.layers.conv2d(inputs=net, filters=128,
                                   kernel_size=[5, 5], padding="same", name="conv3", strides=2,
                                   activation=tf.nn.relu, reuse=reuse)
            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool3")
            # net = tflearn.local_response_normalization(net)
            # net = tflearn.dropout(net, 0.8)

            # net = tf.reshape(net, [-1, (self.config.image_width / 8) * (self.config.image_height / 8) * 128])

            net = tf.layers.flatten(net)
            #tf.reshape(net, [-1, (self.config.image_width / 8) * (self.config.image_height / 8) * 128])

        with tf.variable_scope("spen/fy"):
            nety = yinput

            nety = tf.reshape(nety, shape=(-1, self.config.image_width, self.config.image_height, 2))
            nety = tf.expand_dims(nety[:, :, :, 1],-1)

            nety = tf.layers.conv2d(inputs=nety, filters=8,
                                   kernel_size=[5, 5], strides=1, padding="same", name="conv0",
                                   activation=tf.nn.relu, reuse=reuse)

            nety = tf.layers.conv2d(inputs=nety, filters=8, strides=2,
                                   kernel_size=[5, 5], padding="same", name="conv1",
                                   activation=tf.nn.relu, reuse=reuse)

            nety = tf.layers.conv2d(inputs=nety, filters=16,
                                   kernel_size=[5, 5], padding="same", name="conv2", strides=2,
                                   activation=tf.nn.relu, reuse=reuse)

            nety = tf.layers.flatten(nety)
            # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool3")

            net = tf.concat((net,nety), axis=1)

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

            local_e = tflearn.fully_connected(net, 1, activation='linear',
                                             regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(), bias=False,
                                             reuse=reuse, scope=("fx.fc"))
            #logits = tf.reshape(logits, (-1, self.config.output_num, self.config.dimension))
            #logits = tf.nn.softmax(logits)
            #unipotential = tf.reshape(-tf.log(logits), (-1, self.config.output_num*self.config.dimension))


        return tf.squeeze(local_e)



    def get_energy_horses2(self, xinput=None, yinput=None, embedding=None, reuse=False):
        j = 0
        net = xinput
        with tf.variable_scope("spen/fx"):

            net = tf.reshape(net, shape=(-1, self.config.image_width, self.config.image_height, 3))

            mask = tf.reshape(yinput, shape=(-1, self.config.image_width, self.config.image_width, 2))
            mask = tf.expand_dims(mask[:, :, :, 1], -1)
            net = tf.concat((net, mask), axis=3)

            net = tf.layers.conv2d(inputs=net, filters=16,
                                   kernel_size=[3, 3], strides=1, padding="same", name="conv00",
                                   activation=tf.nn.relu, reuse=reuse)

            net = tf.layers.conv2d(inputs=net, filters=16,
                                   kernel_size=[3, 3], strides=1, padding="same", name="conv01",
                                   activation=tf.nn.relu, reuse=reuse)

            net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool1")

            net = tf.layers.conv2d(inputs=net, filters=32,
                                   kernel_size=[3, 3], strides=1, padding="same", name="conv10",
                                   activation=tf.nn.relu, reuse=reuse)

            net = tf.layers.conv2d(inputs=net, filters=32, strides=1,
                                   kernel_size=[3, 3], padding="same", name="conv11",
                                   activation=tf.nn.relu, reuse=reuse)

            net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool2")

            net = tf.layers.flatten(net)
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

            local_e = tflearn.fully_connected(net, 1, activation='linear',
                                             regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(), bias=False,
                                             reuse=reuse, scope=("fx.fc"))
            #logits = tf.reshape(logits, (-1, self.config.output_num, self.config.dimension))
            #logits = tf.nn.softmax(logits)
            #unipotential = tf.reshape(-tf.log(logits), (-1, self.config.output_num*self.config.dimension))

        with tf.variable_scope("spen/en") as scope:
            #yinput = tf.nn.softmax(tf.reshape(yinput, (-1, self.config.output_num, self.config.dimension)))
            #yinput = tf.reshape(yinput, (-1, self.config.output_num * self.config.dimension))

            #mult = tf.multiply(logits, yinput)
            #local_e = tf.reduce_sum(mult, axis=1)

            with tf.variable_scope(self.config.en_variable_scope) as scope:
                net = yinput
                net = tf.reshape(net, shape=(-1, self.config.image_width, self.config.image_height, 2))

                net = tf.layers.conv2d(inputs=net, filters=16,
                                       kernel_size=[3, 3], strides=1, padding="same", name="conv00y",
                                       activation=tf.nn.relu, reuse=reuse)

                net = tf.layers.conv2d(inputs=net, filters=16,
                                       kernel_size=[3, 3], strides=1, padding="same", name="conv01y",
                                       activation=tf.nn.relu, reuse=reuse)

                net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool1y")

                net = tf.layers.flatten(net)


                j = 0
                for (sz, a) in self.config.en_layer_info:
                    net = tflearn.fully_connected(net, sz,
                                                  weight_decay=self.config.weight_decay,
                                                  weights_init=tfi.variance_scaling(),
                                                  activation=a,
                                                  bias=False,
                                                  reuse=reuse, regularizer='L2',
                                                  scope=("en.h" + str(j)))

                    # net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope=("bn.d" + str(j)))
                    # net = tflearn.activations.softplus(net)
                    # net = tf.log(tf.exp(net) + self.config.temperature)
                    # net = tflearn.dropout(net, 1.0 - self.config.dropout)
                    # net = tf.contrib.layers.layer_norm(net, reuse=reuse, scope=("ln"+str(j)))
                    j = j + 1
                global_e = tf.squeeze(
                    tflearn.fully_connected(net, 1, activation='linear', weight_decay=self.config.weight_decay,
                                            weights_init=tfi.variance_scaling(), bias=False,
                                            reuse=reuse, regularizer='L2',
                                            scope=("en.g")))

            return tf.squeeze(tf.add(local_e, global_e))

            # if reuse:
            #     scope.reuse_variables()
            #
            # init = tf.constant(np.random.normal(0.0,0.0001,size=[self.config.output_num*self.config.dimension,100]), dtype=tf.float32)
            #
            #
            # w1 = tf.get_variable("Interactions", dtype=tf.float32, initializer=init)
            #                  #shape=[self.config.output_num*self.config.dimension,100])
            #
            # #w2 = tf.get_variable("Interactions2", dtype=tf.float32,
            # #                 shape=[100, self.config.output_num*self.config.dimension])
            #
            # W = tf.matmul(w1, w1, transpose_b=True)
            #
            #
            # rel =tf.matmul(tf.matmul(yinput, W), yinput, transpose_b=True)
            # diag = tf.diag_part(rel)
            # global_e = tf.expand_dims(diag,1)
            #
            # print global_e.get_shape().as_list()
            # print local_e.get_shape().as_list()




    def denoise_prediction_network(self, input=None, xinput=None, reuse=False):

        # xinput = xinput * (self.config.dimension-1)
        # yt_ind = self.var_to_indicator(ybatch)
        #           tf.reshape(xinput, (-1, self.config.output_num, self.config.dimension))
        joint_input = tf.concat((input, xinput), axis=1)
        return self.softmax_prediction_network(input=joint_input, xinput=None, reuse=reuse)

    def mlp_prediction_network(self, input=None, xinput=None, reuse=False):

        with tf.variable_scope("pred"):
            with tf.variable_scope("hpred") as scope:
                neth = tflearn.fully_connected(input, 512, activation='relu', regularizer='L2',
                                               weight_decay=self.config.weight_decay,
                                               weights_init=tfi.variance_scaling(),
                                               bias_init=tfi.zeros(), reuse=reuse,
                                               scope=("ph.0"))

            with tf.variable_scope("xpred") as scope:
                netx = tflearn.fully_connected(xinput, 512, activation='relu', regularizer='L2',
                                               weight_decay=self.config.weight_decay,
                                               weights_init=tfi.variance_scaling(),
                                               bias_init=tfi.zeros(), reuse=reuse,
                                               scope=("phx.0"))
                # net = tflearn.dropout(net, 1.0 - self.config.dropout)

            with tf.variable_scope("hpred") as scope:
                net = tf.concat((neth, netx), axis=1)
                net = tflearn.fully_connected(net, 512, activation='relu', regularizer='L2',
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              bias_init=tfi.zeros(), reuse=reuse,
                                              scope=("ph.1"))
                # net = tflearn.dropout(net, 1.0 - self.config.dropout)

                net = tflearn.fully_connected(net, self.config.output_num * self.config.dimension,
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              bias_init=tfi.zeros(), reuse=reuse,
                                              regularizer='L2',
                                              scope=("ph.fc"))
        return tf.nn.sigmoid(net)

    def energy_image_denoise(self, xinput=None, yinput=None, embedding=None, reuse=False):
        with tf.variable_scope(self.config.spen_variable_scope):
            # pred = self.mlp_prediction_network(yinput, xinput=xinput, reuse=False)
            local_e = -tf.reduce_sum(tf.square(xinput - yinput), 1)
            with tf.variable_scope(self.config.en_variable_scope) as scope:
                net = yinput
                j = 0

                for (sz, a) in self.config.en_layer_info:
                    net = tflearn.fully_connected(net, sz, activation=a,
                                                  weight_decay=self.config.weight_decay,
                                                  weights_init=tfi.variance_scaling(),
                                                  bias_init=tfi.zeros(), reuse=reuse,
                                                  scope=("en.h" + str(j)))
                    j = j + 1

                global_e = tflearn.fully_connected(net, 1, activation='linear', weight_decay=self.config.weight_decay,
                                                   weights_init=tfi.zeros(),
                                                   bias_init=tfi.zeros(), reuse=reuse,
                                                   scope=("en.h" + str(j)))

        net = local_e + 0.1 * global_e

        return tf.squeeze(net)

    def simple_prediction_network(self, input=None, xinput=None, reuse=False):
        net = tf.concat((input, xinput), axis=1)
        with tf.variable_scope("pred") as scope:
            net = tflearn.fully_connected(net, 1000, regularizer='L2',
                                          weight_decay=self.config.weight_decay,
                                          weights_init=tfi.variance_scaling(),
                                          bias_init=tfi.zeros(), reuse=reuse,
                                          scope=("ph.1"))

            # net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope=("bn1"))
            net = tf.nn.relu(net)
            net = tflearn.layers.dropout(net, 1 - self.config.dropout)
            net = tflearn.fully_connected(net, self.config.output_num * self.config.dimension,
                                          weight_decay=self.config.weight_decay,
                                          weights_init=tfi.variance_scaling(),
                                          bias_init=tfi.zeros(), reuse=reuse,
                                          regularizer='L2',
                                          scope=("ph.2"))
            # net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope=("bn2"))
        return tf.nn.sigmoid(net)

    def get_energy_card(self, xinput=None, yinput=None, embedding=None, reuse=False):
        output_size = yinput.get_shape().as_list()[-1]
        with tf.variable_scope(self.config.spen_variable_scope):
            with tf.variable_scope(self.config.fx_variable_scope) as scope:
                logits = self.get_feature_net_mlp(xinput, output_size, reuse=reuse)
                mult = tf.multiply(logits, yinput)
                local_e = tflearn.fully_connected(mult, 1, activation='linear', regularizer='L2',
                                                  weight_decay=self.config.weight_decay,
                                                  weights_init=tfi.variance_scaling(),
                                                  bias=False,
                                                  bias_init=tfi.zeros(), reuse=reuse, scope=("en.l"))
            with tf.variable_scope(self.config.en_variable_scope) as scope:
                j = 0
                net = yinput
                for (sz, a) in self.config.en_layer_info:
                    net = tflearn.fully_connected(net, sz,
                                                  weight_decay=self.config.weight_decay,
                                                  weights_init=tfi.variance_scaling(),
                                                  bias_init=tfi.zeros(), reuse=reuse, regularizer='L2',
                                                  scope=("en.h" + str(j)))

                    # net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope=("bn.f" + str(j)))
                    # net = tflearn.activations.softplus(net)
                    net = tf.log(tf.exp(net) + 1.0)
                    net = tflearn.dropout(net, 1.0 - self.config.dropout)
                    j = j + 1
                global_e = tflearn.fully_connected(net, 1, activation='linear', weight_decay=self.config.weight_decay,
                                                   weights_init=tfi.zeros(), bias=False,
                                                   reuse=reuse, regularizer='L2',
                                                   scope=("en.g"))

                card = tf.reduce_sum(yinput, 1)

        return tf.squeeze(local_e + global_e + card)

    def simple_hyper_network(self, input=None, xinput=None, reuse=False):
        net = xinput

        with tf.variable_scope("pred") as scope:
            net = tflearn.fully_connected(net, self.config.hidden_num, regularizer='L2',
                                          weight_decay=self.config.weight_decay,
                                          weights_init=tfi.variance_scaling(),
                                          bias_init=tfi.zeros(), reuse=reuse,
                                          activation='linear',
                                          scope=("ph.1"))

            mult = tf.multiply(net, input)

            net = tf.nn.relu(mult)

            output = tflearn.fully_connected(net, self.config.output_num * self.config.dimension, regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(),
                                             bias_init=tfi.zeros(), reuse=reuse,
                                             activation='sigmoid',
                                             scope=("ph.o"))

        return output

    def another_hyper_network(self, input=None, xinput=None, reuse=False):
        def fn(acc, elm):
            xinput, weights = elm
            xinput = tf.expand_dims(xinput, 1)
            sum = tf.matmul(weights, xinput)
            # print (tf.shape(sum).as_list())
            return sum

        with tf.variable_scope("pred") as scope:
            if reuse:
                scope.reuse_variables()
            size = self.config.hidden_num / self.config.input_num
            bias = tf.get_variable("b", shape=(size))
            weight = tf.reshape(input, (-1, size, self.config.input_num))
            v = tf.scan(fn, (xinput, weight), tf.zeros((size, 1)))
            v = tf.reshape(v, (-1, self.config.hidden_num / self.config.input_num))
            net = tf.nn.relu(tf.nn.bias_add(v, bias))
            output = tflearn.fully_connected(net, self.config.output_num * self.config.dimension, regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(),
                                             bias_init=tfi.zeros(), reuse=reuse,
                                             activation='sigmoid',
                                             scope=("ph.o"))

            return output

    def get_energy_dep(self, xinput=None, yinput=None, embedding=None, reuse=False):
        h = tf.expand_dims(yinput, axis=-1)

        with tf.variable_scope(self.config.spen_variable_scope):
            with tf.variable_scope(self.config.fx_variable_scope) as scope:
                net = xinput
                j = 0
                for (sz, a) in self.config.layer_info:
                    net = tflearn.fully_connected(net, sz,
                                                  weight_decay=self.config.weight_decay,
                                                  weights_init=tfi.variance_scaling(),
                                                  bias_init=tfi.zeros(), regularizer='L2', reuse=reuse,
                                                  scope=("fx.h" + str(j)))
                    # net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope=("bn.f" + str(j)))
                    net = tflearn.activations.relu(net)
                    # net = tflearn.dropout(net, 1.0 - self.config.dropout)
                    j = j + 1

                logits = tflearn.fully_connected(net, self.config.hidden_num * 2, regularizer='L2',
                                                 weight_decay=self.config.weight_decay,
                                                 weights_init=tfi.variance_scaling(),
                                                 bias_init=tfi.zeros(), reuse=reuse,
                                                 activation='linear',
                                                 scope=("fc"))

                c = tf.expand_dims(logits[:, :self.config.hidden_num], axis=-1)
                b = tf.expand_dims(logits[:, self.config.hidden_num:], axis=-1)
                A = tf.matmul(c, c, transpose_b=True)
                print(("A:", A.get_shape().as_list()))

                yTA = tf.matmul(h, A, transpose_a=True)

                print(("yTA:", yTA.get_shape().as_list()))

                yTAy = tf.matmul(yTA, h)
                print(("yTAy:", yTAy.get_shape().as_list()))

                global_e = -yTAy
                local_e = - tf.matmul(b, h, transpose_a=True)

        return tf.squeeze(global_e + local_e)

    def get_energy_dep2(self, xinput=None, yinput=None, embedding=None, reuse=False):

        with tf.variable_scope(self.config.spen_variable_scope):
            with tf.variable_scope(self.config.fx_variable_scope) as scope:
                net = xinput
                j = 0
                for (sz, a) in self.config.layer_info:
                    net = tflearn.fully_connected(net, sz,
                                                  weight_decay=self.config.weight_decay,
                                                  weights_init=tfi.variance_scaling(),
                                                  bias_init=tfi.zeros(), regularizer='L2', reuse=reuse,
                                                  scope=("fx.h" + str(j)))
                    net = tflearn.activations.relu(net)
                    j = j + 1

                logits = tflearn.fully_connected(net, self.config.hidden_num * 2, regularizer='L2',
                                                 weight_decay=self.config.weight_decay,
                                                 weights_init=tfi.variance_scaling(),
                                                 bias_init=tfi.zeros(), reuse=reuse,
                                                 activation='sigmoid',
                                                 scope=("fc"))

                b = tf.multiply(logits[:, self.config.hidden_num:], yinput)

                local_e = tflearn.fully_connected(b, 1, regularizer='L2',
                                                  weight_decay=self.config.weight_decay,
                                                  weights_init=tfi.variance_scaling(),
                                                  bias=False, reuse=reuse,
                                                  activation='linear',
                                                  scope=("en.l"))

                d = tf.multiply(logits[:, :self.config.hidden_num], yinput)
                net = d
                for (sz, a) in self.config.en_layer_info:
                    net = tflearn.fully_connected(net, sz,
                                                  weight_decay=self.config.weight_decay,
                                                  weights_init=tfi.variance_scaling(),
                                                  bias_init=tfi.zeros(), reuse=reuse, regularizer='L2',
                                                  scope=("en.h" + str(j)))

                    net = tf.log(tf.exp(net) + 1.0)
                    j = j + 1

                global_e = tflearn.fully_connected(net, 1, regularizer='L2',
                                                   weight_decay=self.config.weight_decay,
                                                   weights_init=tfi.variance_scaling(),
                                                   reuse=reuse,
                                                   activation='linear',
                                                   bias=False,
                                                   scope=("en.g"))

        return tf.squeeze(global_e + local_e)

    def top_hyper_network(self, input=None, xinput=None, reuse=False):
        def fn(acc, elm):
            inp, weights = elm
            inp = tf.expand_dims(inp, 1)
            sum = tf.matmul(weights, inp)
            # print (tf.shape(sum).as_list())
            return sum

        with tf.variable_scope("pred") as scope:
            if reuse:
                scope.reuse_variables()
            net = xinput
            net = tflearn.fully_connected(net, 1000, regularizer='L2',
                                          weight_decay=self.config.weight_decay,
                                          weights_init=tfi.variance_scaling(),
                                          bias_init=tfi.zeros(), reuse=reuse,
                                          activation='relu',
                                          scope=("ph.1"))

            size = 100
            bias = tf.get_variable("b", shape=(size))
            weight = tf.reshape(input, (-1, size, 1000))
            v = tf.scan(fn, (net, weight), tf.zeros((size, 1)))
            v = tf.reshape(v, (-1, size))
            net = tf.nn.relu(tf.nn.bias_add(v, bias))

            output = tflearn.fully_connected(net, self.config.output_num * self.config.dimension, regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(),
                                             bias_init=tfi.zeros(), reuse=reuse,
                                             activation='sigmoid',
                                             scope=("ph.o"))

            return output

    def mixture_prediction_network(self, input=None, xinput=None, reuse=False):
        net = xinput
        # with tf.variable_scope("pred") as scope:
        #   net = tf.reshape(net, shape=(-1, self.config.image_width, self.config.image_height, 3))
        #
        #   net = tf.layers.conv2d(inputs=net, filters=8,
        #                        kernel_size=[5, 5], padding="same", name="conv1",
        #                        activation=tf.nn.relu, reuse=reuse)
        # #  net = tf.layers.batch_normalization()
        #   net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool1")
        #   net = tflearn.local_response_normalization(net)
        # # net = tflearn.dropout(net, 0.8)
        #
        #   net = tf.layers.conv2d(inputs=net, filters=16,
        #                        kernel_size=[5, 5], padding="same", name="conv2",
        #                        activation=tf.nn.relu, reuse=reuse)
        #
        #   net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name="pool2")
        #   net = tflearn.local_response_normalization(net)
        # # net = tflearn.dropout(net, 0.8)
        #
        # # net = tf.layers.conv2d(inputs=net, filters=32,
        # #                       kernel_size=[5, 5], padding="same", name="conv3",
        # #                       activation=tf.nn.relu, reuse=reuse)
        #
        # # net = tflearn.local_response_normalization(net)
        # # net = tflearn.dropout(net, 0.8)
        #
        # net = tf.reshape(net, [-1, (self.config.image_width / 4) * (self.config.image_height / 4) * 16])

        # input = tf.nn.softmax(input)
        # net = tflearn.layers.dropout(net, 0.8)
        j = 0
        with tf.variable_scope("pred") as scope:
            for (sz, a) in self.config.pred_layer_info:
                net = tflearn.fully_connected(net, sz, regularizer='L2',
                                              activation=a,
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              bias_init=tfi.zeros(), reuse=reuse,
                                              scope=("ph." + str(j)))
                net = tflearn.layers.dropout(net, 1 - self.config.dropout)
                j = j + 1

            net = tflearn.fully_connected(net, self.config.hidden_num * self.config.output_num * self.config.dimension,
                                          activation='linear',
                                          weight_decay=self.config.weight_decay,
                                          weights_init=tfi.variance_scaling(),
                                          bias=False,
                                          reuse=reuse,
                                          regularizer='L2',
                                          scope=("ph.fc"))
            if self.config.dimension == 1:
                net = tf.reshape(net, (-1, self.config.hidden_num, self.config.output_num))
                net = tf.nn.sigmoid(net)
            else:
                net = tf.reshape(net, (-1, self.config.hidden_num, self.config.output_num, self.config.dimension))
                net = tf.nn.softmax(net, dim=-1)

            net = tf.reshape(net, (-1, self.config.hidden_num, self.config.output_num * self.config.dimension))

        def fn(acc, elm):
            h, frame = elm
            h = tf.expand_dims(h, 1)
            sum = tf.matmul(h, frame, transpose_a=True)
            # print (tf.shape(sum).as_list())
            return sum

        with tf.variable_scope("mixture") as scope:
            if reuse:
                scope.reuse_variables()

            v = tf.scan(fn, (input, net), tf.zeros((1, self.config.output_num * self.config.dimension)))

        return tf.squeeze(v)

    def loss_mlp(self, ytrue=None, ypred=None, reuse=False):
        mult = tf.multiply(ytrue, ypred)
        all = tf.concat((mult, ytrue, ypred), 1)
        net = all
        j = 0
        net = tflearn.fully_connected(net, 500, regularizer='L2',
                                      activation='relu',
                                      weight_decay=self.config.weight_decay,
                                      weights_init=tfi.variance_scaling(),
                                      bias_init=tfi.zeros(), reuse=reuse,
                                      scope=("lh." + str(j)))

        net = tflearn.fully_connected(net, 1, regularizer='L2',
                                      activation='sigmoid',
                                      weight_decay=self.config.weight_decay,
                                      weights_init=tfi.variance_scaling(),
                                      bias_init=tfi.zeros(), reuse=reuse,
                                      scope=("ll." + str(j)))
        return net

    def get_energy_text_cnn(self, xinput=None, yinput=None, embedding=None, reuse=False):
        xinput = tf.cast(xinput, tf.int32)
        xinput = tf.nn.embedding_lookup(embedding, xinput)
        yinput = tf.reshape(yinput, (-1, self.config.output_num, self.config.dimension))
        xinput_x = xinput #tf.expand_dims(xinput, -1)
        yinput_x = yinput # tf.expand_dims(yinput, -1)

        pooled_outputs = []

        print("xinput", xinput_x.get_shape().as_list())
        print("yinput", yinput_x.get_shape().as_list())

        net = tf.concat((xinput_x, yinput_x), axis=-1)
        net = tf.expand_dims(net, -1)
        print("input", net.get_shape().as_list())
        with tf.variable_scope("spen/fx"):
            net = tf.layers.conv2d(inputs=net, filters= 200,
                               kernel_size=[3, 64], strides=1, padding="SAME", name="conv00",
                               activation=tf.nn.relu, reuse=reuse)

            #net = tf.layers.conv2d(inputs=net, filters=100,
            #                       kernel_size=[3, 3], strides=1, padding="SAME", name="conv01",
            #                       activation=tf.nn.relu, reuse=reuse)
            # net = tf.layers.conv2d(inputs=net, filters=20,
            #                        kernel_size=[3, 64], strides=1, padding="VALID", name="conv01",
            #                        activation=tf.nn.relu, reuse=reuse)

            net = tf.layers.max_pooling2d(inputs=net, pool_size=[self.config.output_num,1], strides=1, name="pool1", padding="VALID")

            net = tf.layers.flatten(net)

            j = 0
            for (sz, a) in self.config.layer_info:
                net = tflearn.fully_connected(net, sz,
                                          weight_decay=self.config.weight_decay,
                                          weights_init=tfi.variance_scaling(),
                                          activation=a,
                                          bias_init=tfi.zeros(), regularizer='L2', reuse=reuse,
                                          scope=("fx.h" + str(j)))
                net = tflearn.dropout(net, 1.0 - self.config.dropout)

                j = j + 1

            output = tflearn.fully_connected(net, 1, activation='linear',
                                         regularizer='L2',
                                         weight_decay=self.config.weight_decay,
                                         weights_init=tfi.variance_scaling(), bias=False,
                                         reuse=reuse, scope=("fx.fc"))

        return tf.squeeze(output)


import tensorflow as tf
import tflearn
import tflearn.initializations as tfi

class EnergyModel:
  def __init__(self,config):
    self.config = config

  def get_feature_net_rnn(self, xinput, reuse=False):
    with tf.variable_scope("bi-lstm") as scope:
      if reuse:
        scope.reuse_variables()

      cell_fw = tf.contrib.rnn.LSTMCell(self.config.lstm_hidden_size, reuse=reuse)
      cell_bw = tf.contrib.rnn.LSTMCell(self.config.lstm_hidden_size, reuse=reuse)
      (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, xinput,  dtype=tf.float32)
      output = tf.concat([output_fw,  output_bw], axis=-1)
      #output = tf.contrib.layers.layer_norm(output, reuse=reuse, scope=("ln0"))
      #output = tf.nn.dropout(output, 1- self.config.dropout)
    nsteps = tf.shape(output)[1]

    with tf.variable_scope("proj") as scope:
      if reuse:
        scope.reuse_variables()
      W = tf.get_variable("PW", dtype=tf.float32,
                          shape=[2 * self.config.lstm_hidden_size, self.config.dimension])

      b = tf.get_variable("Pb", shape=[self.config.dimension],
                          dtype=tf.float32, initializer=tf.zeros_initializer())

      output = tf.reshape(output, [-1, 2 * self.config.lstm_hidden_size])
      pred = tf.matmul(output, W) + b
      logits = tf.reshape(pred, [-1, nsteps*self.config.dimension])

    return logits



  def get_feature_net_mlp(self, xinput, reuse=False):
    net = xinput
    j = 0
    for (sz, a) in self.config.layer_info:
      net = tflearn.fully_connected(net, sz,
                                    weight_decay=self.config.weight_decay,
                                    weights_init=tfi.variance_scaling(),
                                    bias_init=tfi.zeros(), reuse=reuse, scope=("fx.h" + str(j)))
      net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope=("bn.f" + str(j)))
      net = tflearn.activations.relu(net)
      #net = tflearn.dropout(net, 1.0 - self.dropout_ph)
      j = j + 1
    sz = self.config.output_num
    logits = tflearn.fully_connected(net, sz, activation='linear', weight_decay=self.config.weight_decay,
                                     weights_init=tfi.variance_scaling(),
                                     bias_init=tfi.zeros(), reuse=reuse, scope=("fx.h" + str(j)))
    return logits

  def get_energy_mlp(self, xinput, yinput, reuse=False):
    with tf.variable_scope(self.config.spen_variable_scope):
      with tf.variable_scope(self.config.fx_variable_scope) as scope:
        logits = self.get_feature_net_mlp(xinput, reuse=reuse)
        mult = tf.multiply(logits, yinput)
        local_e = tflearn.fully_connected(mult, 1, activation='linear', weight_decay=self.config.weight_decay,
                                           weights_init=tfi.variance_scaling(),
                                           bias=None,
                                           bias_init=tfi.zeros(), reuse=reuse, scope=("fx3.b0"))
      with tf.variable_scope(self.config.en_variable_scope) as scope:
        j = 0
        net = yinput
        for (sz, a) in self.config.en_layer_info:
          net = tflearn.fully_connected(net, sz,
                                        weight_decay=self.config.weight_decay,
                                        weights_init=tfi.variance_scaling(),
                                        bias_init=tfi.zeros(), reuse=reuse,
                                        scope=("en.h" + str(j)))
          #net = tflearn.dropout(net, 1.0 - self.config.dropout)
          #net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope=("bn.f" + str(j)))
          net = tflearn.activations.softplus(net)
          j = j + 1
        global_e = tflearn.fully_connected(net, 1, activation='linear', weight_decay=self.config.weight_decay,
                                           weights_init=tfi.zeros(),
                                           bias_init=tfi.zeros(), reuse=reuse,
                                           scope=("en.h" + str(j)))

    return tf.squeeze(local_e + global_e)

  def get_energy_mlp_emb(self, xinput, yinput, reuse=False):
    xinput = tf.cast(xinput, tf.int32)
    xinput = tf.nn.embedding_lookup(self.embedding, xinput)
    return self.get_energy_mlp(xinput, yinput, reuse)


  def get_energy_rnn_mlp_emb(self, xinput, yinput, reuse=False):
    xinput = tf.cast(xinput, tf.int32)
    xinput = tf.nn.embedding_lookup(self.embedding, xinput)
    with tf.variable_scope(self.config.spen_variable_scope):
      with tf.variable_scope(self.config.fx_variable_scope) as scope:
        logits2 = self.get_feature_net_rnn(xinput, reuse=reuse)
        logits3 = self.get_feature_net_mlp(xinput, reuse=reuse)  #+ logits2


        #mult_ = tf.multiply(logits, yinput)
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


        #local_e = tflearn.fully_connected(mult_, 1, activation='linear', weight_decay=self.config.weight_decay,
        #                                  weights_init=tfi.variance_scaling(),
        #                                  bias=None,
        #                                  bias_init=tfi.zeros(), reuse=reuse, scope=("fx.b0"))

      j = 0

      with tf.variable_scope(self.config.en_variable_scope) as scope:
        net = yinput
        j = 0

        for (sz, a) in self.config.en_layer_info:
          # std = np.sqrt(2.0) / np.sqrt(sz)
          net = tflearn.fully_connected(net, sz,activation=a,
                                        weight_decay=self.config.weight_decay,
                                        weights_init=tfi.variance_scaling(),
                                        bias_init=tfi.zeros(), reuse=reuse,
                                        scope=("en.h" + str(j)))
        #  net = tflearn.activations.relu(net)
          #net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope=("bn.en" + str(j)))

          j = j + 1


        global_e = tflearn.fully_connected(net, 1, activation='linear', weight_decay=self.config.weight_decay,
                                           weights_init=tfi.zeros(),
                                           bias_init=tfi.zeros(), reuse=reuse,
                                           scope=("en.h" + str(j)))
        # en = tflearn.layers.normalization.batch_normalization(en, reuse=ru, scope=("en." + str(j)))


        if reuse:
          scope.reuse_variables()

        net = global_e + local_e3 + local_e2#+ local_e + local_e3

      return tf.squeeze(net)

  def get_feature_net_cnn(self, reuse=False):
    net = self.x
    netw = tf.nn.embedding_lookup(self.W, net)
    net_expanded = tf.expand_dims(netw, -1)
    pooled_outputs = []
    for i, filter_size in enumerate(self.config.filter_sizes):
      with tf.variable_scope("max-pooling") as scope:
        if reuse:
          scope.reuse_variables()
        filter_shape = [filter_size, self.embedding_size, 1, self.config.num_filters]
        W = tf.get_variable(initializer=tf.truncated_normal(filter_shape, stddev=0.1), name=("W"+ ("-conv-maxpool-%s" % filter_size)))
        b = tf.get_variable(initializer=tf.constant(0.1, shape=[self.config.num_filters]), name=("b" + ("-conv-maxpool-%s" % filter_size)))
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
    h_pool = tf.concat(pooled_outputs,3)
    net = tf.reshape(h_pool, [-1, num_filters_total])
    net = tflearn.dropout(net, 1.0 - self.config.dropout)
    j=0
    net2 = netw
    sz = self.config.output_num

    logits = tflearn.fully_connected(net, sz, activation='linear', weight_decay=self.config.weight_decay,
                                     weights_init=tfi.variance_scaling(),
                                     bias_init=tfi.zeros(), reuse=reuse, scope=("fx.h-cnn" + str(j)))
    return logits



  def softmax_prediction_network(self, hinput, reuse):
    net = hinput
    with tf.variable_scope(self.config.spen_variable_scope):
      with tf.variable_scope("pred") as scope:
        net = tflearn.fully_connected(net, 1000, activation='relu',
                                      weight_decay=self.config.weight_decay,
                                      weights_init=tfi.variance_scaling(),
                                      bias_init=tfi.zeros(), reuse=reuse,
                                      scope=("ph.0"))
        net = tflearn.fully_connected(net, self.config.output_num, activation='linear',
                                      weight_decay=self.config.weight_decay,
                                      weights_init=tfi.variance_scaling(),
                                      bias_init=tfi.zeros(), reuse=reuse,
                                      scope=("ph.1"))

    cat_output = tf.reshape(net, (-1, self.config.output_num, self.config.dimension))
    return tf.nn.softmax(cat_output, dim=2)





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
    self.y = tf.placeholder(tf.float32, shape=[None, config.output_num , self.config.dimension], name="OutputX")


  def init_embedding(self, embedding):
    self.sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: embedding})
    return self

  def init(self):
    init_op = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(init_op)
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
    l = tf.reduce_mean( tf.square(yt-yp))
    return l

  def biased_loss(self, yt, yp):
    l = -tf.reduce_sum ((tf.reduce_sum(yt * tf.log(tf.maximum(yp, 1e-20)), 1) \
       + tf.reduce_sum((1. - yt) * tf.log(tf.maximum(1. - yp , 1e-20)), 1)) )
    yp = tf.reshape(yp, [-1, self.config.output_num, self.config.dimension])
    yp_zeros = yp[:, :, 0]
    yp_ones = yp[:,:, 1]
    return l + 1.2*(tf.reduce_sum(yp_zeros) - tf.reduce_sum(yp_ones))


  def get_feature_net_mlp(self, xinput, output_num, reuse=False):
    print(output_num)

    net = xinput
    j = 0
    for (sz, a) in self.config.layer_info:
      print (sz, a)
      net = tflearn.fully_connected(net, sz,
              weight_decay=self.config.weight_decay,
              #weights_init=tfi.variance_scaling(,
              bias_init=tfi.zeros(), regularizer='L2', reuse=reuse, scope=("fx.h" + str(j)))
      net = tflearn.activations.relu(net)
      #net = tflearn.dropout(net, 1.0 - self.config.dropout)
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
    print("xinput",  xinput)
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

  def softmax_prediction_network(self, xinput=None, reuse=False):
    net = xinput
    j = 0
    with tf.variable_scope("pred") as scope:
      for (sz, a) in self.config.pred_layer_info:
        print (sz,a)
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

    cat_output = tf.reshape(net, (-1, self.config.output_num, self.config.dimension))

    return tf.nn.softmax(cat_output, dim=2)

  def cnn_prediction_network(self, xinput=None, reuse=False):

    #input = tf.concat((xinput, yinput), axis=1)
    net = xinput
    j = 0
    with tf.variable_scope("pred"):
      net = tf.reshape(net, shape=(-1, self.config.image_width , self.config.image_height,1) )

      for (nf, fs, st) in self.config.cnn_layer_info:
        net = tflearn.conv_2d(net, nb_filter=nf, filter_size=fs, strides=st,
                              padding="same", scope=("conv" + str(j)), activation=tf.nn.relu, reuse=reuse)
        #net = tflearn.max_pool_2d(net, kernel_size=[2,2], strides=2)
        #net = tflearn.batch_normalization(net, scope=("bn"+ str(j)), reuse=reuse)
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


  def map_predict(self, xinput=None, train=False):
    tflearn.is_training(train, self.sess)
    yp = self.sess.run(self.yp, feed_dict={self.x: xinput})
    if self.config.loglevel > 1:
      print (yp)
    return np.argmax(yp,2)

  def pred_variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


  def createOptimizer(self):
    self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)

  def construct(self):
    self.yp = self.softmax_prediction_network(self.x)
    self.objective = self.get_loss(self.y, self.yp) + self.config.l2_penalty * self.get_l2_loss()
    self.train_step = self.optimizer.minimize(self.objective)
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
    yt_ind = self.var_to_indicator(ybatch)
    yt_ind = np.reshape(yt_ind, (-1, self.config.output_num, self.config.dimension))
    feeddic = {self.x:xbatch, self.y: yt_ind,
               self.learning_rate_ph:self.config.learning_rate,
               self.dropout_ph: self.config.dropout}
    _, o = self.sess.run([self.train_step, self.objective], feed_dict=feeddic)
    if verbose > 0:
      print (self.train_iter ,o)
    return o

  def var_to_indicator(self, vd):
    size = np.shape(vd)
    cat = np.zeros((size[0], self.config.output_num, self.config.dimension))
    for i in range(size[0]):
      for j in range(self.config.output_num):
        k = vd[i, j]
        cat[i, j, int(k)] = 1
    return np.reshape(cat, (size[0], self.config.output_num, self.config.dimension))
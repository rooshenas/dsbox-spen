
import tensorflow as tf
import numpy as np

import tflearn
import tflearn.initializations as tfi
import random
from enum import Enum

class Config:
  def __init__(self):
    self.dropout = 0.0
    self.layer_info = None
    self.en_layer_info = None
    self.weight_decay = 0.001
    self.l2_penalty = 0.0
    self.en_variable_scope = "en"
    self.fx_variable_scope = "fx"
    self.spen_variable_scope = "spen"
    self.inf_rate = 0.1

    self.learning_rate = 0.001
    self.inf_iter = 10
    self.dropout = 0.0
    self.train_iter = 10
    self.batch_size = 100
    self.dimension = 2
    self.filter_sizes = [2,3,4,5]
    self.num_filters = 10
    self.num_samples = 10
    self.margin_weight = 100.0
    self.exploration = 0.0
    self.lstm_hidden_size = 100
    self.vocabulary_size = 20608
    self.embedding_size = 100
    self.sequence_length = 118
    self.hidden_num = 100
    self.input_num = 0
    self.output_num = 0



class InfInit(Enum):
  Random_Initialization = 1
  GT_Initialization = 2
  Zero_Initialization = 3

class TrainingType:
  Value_Matching = 1
  SSVM = 2
  Rank_Based = 3


class SPEN:
  def __init__(self,config):
    self.config = config
    self.x = tf.placeholder(tf.float32, shape=[None, self.config.input_num])
    self.learning_rate_ph = tf.placeholder(tf.float32, shape=[None])
    self.dropout_ph = tf.placeholder(tf.float32, shape=[None])
    return self


  def init(self):
    init_op = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(init_op)
    return self

  def init_embedding(self, embedding):
    self.sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: embedding})
    return self




  def print_vars(self):
    for v in self.spen_variables():
      print(v)

  def spen_variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen")

  def energy_variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen/en")

  def fnet_variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen/fx")

  def ssvm_training(self):
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


  def construct(self, training_type = TrainingType.SSVM ):
    if training_type == TrainingType.SSVM:
      return self.ssvm_training()
    elif training_type == TrainingType.Rank_Based:
      return self.rank_training()
    else:
      raise NotImplementedError




  def project_simplex_norm(self, yd):
    dim = self.config.dimension
    y_min = np.min(yd, axis=2)
    y_min_all = np.reshape(np.repeat(y_min, dim), (-1, self.config.output_num / dim, dim))
    yd_pos = yd - y_min_all
    yd_sum = np.reshape(np.repeat(np.sum(yd_pos, 2), dim), (-1, self.config.output_num / dim, dim))
    yd_norm = np.divide(yd_pos, yd_sum)
    return yd_norm

  def project_simplex(self, yd):
    return self.project_simplex_norm(yd)

  def get_energy(self, reuse=False):
    raise NotImplementedError


  def rank_training(self):
    self.h1 = tf.placeholder(tf.float32, shape=[None, self.config.hidden_num])
    self.h2 = tf.placeholder(tf.float32, shape=[None, self.config.hidden_num])
    self.margin_weight_ph = tf.placeholder(tf.float32, shape=[None])


    self.yp1 = self.prediction_net(self.h1)
    self.yp2 = self.prediction_net(self.h2, reuse=True)

    flat_yp1 =tf.reshape(self.yp1, shape=(-1, self.config.output_num*self.config.dimension))
    flat_yp2 = tf.reshape(self.yp2, shape=(-1, self.config.output_num*self.config.dimension))

    self.spen_y1 = self.get_energy(self.x, flat_yp1)
    self.spen_y2 = self.get_energy(self.x, flat_yp2, reuse=True)
    self.spen_ygradient2 = tf.gradients(self.spen_y2, self.h2)[0]

    self.ce1 = -tf.reduce_sum(self.y1 * tf.log(tf.maximum(flat_yp1, 1e-20)), 1)
    self.ce2 = -tf.reduce_sum(self.y1 * tf.log(tf.maximum(flat_yp2, 1e-20)), 1)

    vloss = self.get_l2_loss()

    obj1 = tf.reduce_sum(tf.maximum((self.ce2 - self.ce1) * self.margin_weight_ph - self.spen_y1 + self.spen_y2, 0.0))
    self.v1_sum = tf.reduce_sum(self.ce1)
    self.v2_sum = tf.reduce_sum(self.ce2)
    self.e1_sum = tf.reduce_sum(self.spen_y1)
    self.e2_sum = tf.reduce_sum(self.spen_y2)
    self.objective = obj1 + self.config.l2_penalty * vloss  # + obj2
    self.num_update = tf.reduce_sum(
      tf.cast((self.ce1 - self.ce2) * self.margin_weight_ph >= (self.spen_y1 - self.spen_y2), tf.float32))

    self.train_step = tf.train.AdamOptimizer(self.learning_rate_ph).minimize(self.objective, var_list=self.spen_variables())

    return self

  def inference(self, xd, yt=None, inf_iter = None, train=True, ascent=True, initialization: InfInit = InfInit.Random_Initialization,):
    """
      ARGS:
        xd: Input Tensor
        yt: Ground Truth Output

      RETURNS:
        An array of Tensor of shape (-1, output_num, dimension)

    """
    if inf_iter is None:
      inf_iter = self.config.inf_rate
    tflearn.is_training(is_training=train, session=self.sess)
    bs = np.shape(xd)[0]

    if initialization == InfInit.Random_Initialization:
      hd = np.random.uniform(0,1.0, (bs, self.config.hidden_dimension))
    else:
      raise NotImplementedError("Other initialization methods are not supported.")

    i=0
    h_a = []
    while i < inf_iter:
      g = self.sess.run(self.inf_gradient, feed_dict={self.x:xd, self.h:hd, self.dropout_ph: self.config.dropout})
      if ascent:
        hd = hd + self.config.inf_rate * (g)
      else:
        hd = hd - self.config.inf_rate * (g)
      h_a.append(hd)

    return np.array(hd)


  def set_train_iter(self, iter):
    self._train_iter = iter

  def get_train_iter(self):
    return self._train_iter

  def prediction_net(self, hidden_vars=None, reuse=False ):
    raise NotImplementedError

  def get_l2_loss(self):
    loss = 0.0
    en_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.config.spen_variable_scope)
    for v in en_vars:
      loss += tf.nn.l2_loss(v)
    return loss

  def var_to_indicator(self, vd):
    size = np.shape(vd)
    cat = np.zeros((size[0], self.config.output_num, self.config.dimension))
    for i in range(size[0]):
      for j in range(self.config.output_num):
        k = vd[i, j]
        cat[i, j, int(k)] = 1
    return np.reshape(cat, (size[0], self.config.output_num * self.config.dimension))

  def indicator_to_var(self, id):
    size = np.shape(id)
    y_cat_indicator = np.reshape(id, (size[0], self.config.output_num, self.config.dimension))
    y_m = np.argmax(y_cat_indicator, 2)
    return y_m




  def soft_predict(self, xd, train=False, inf_iter=None):
    hd = self.hpredict(xd, train=train, inf_iter=inf_iter)
    yp = self.sess(self.yp1, feed_dict={self.y1: hd, self.dropout_ph: self.config.dropout})
    return yp

  def map_predict(self, xd, train=False, inf_iter=None):
    yp = self.soft_predict(xd, train=train, inf_iter=inf_iter)
    return np.argmax(yp, 2)

  def hpredict(self, xd=None, inf_iter=None, train=False ):
    self.inf_objective = self.spen_y2
    self.inf_gradient = self.spen_ygradient2
    h_a = self.inference(xd, inf_iter=inf_iter, train=train)
    return h_a[-1]


  def train_batch(self, xbatch=None, ybatch=None):
    raise NotImplementedError

  def train_rank_supervised_batch(self, xbatch, ybatch):
    yp = self.soft_predict(xd=xbatch, train=True, inf_iter=10)
    yp_flat = tf.reshape(yp, shape=(-1, self.config.output_num* self.config.dimension))
    yd_flat = self.var_to_indicator(ybatch)
    feeddic={
      self.x: xbatch,
      self.y1: yd_flat,
      self.y2: yp_flat,
      self.learning_rate_ph: self.config.learning_rate,
      self.dropout_ph: self.config.dropout,
      self.margin_weight_ph: self.config.margin_weight
    }
    _, obj = self.sess.run([self.train_step, self.objective], feed_dict=feeddic )
    return obj


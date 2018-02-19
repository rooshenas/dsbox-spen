
import tensorflow as tf
import numpy as np

import tflearn
import tflearn.initializations as tfi
from enum import Enum


class InfInit(Enum):
  Random_Initialization = 1
  GT_Initialization = 2
  Zero_Initialization = 3

class TrainingType(Enum):
  Value_Matching = 1
  SSVM = 2
  Rank_Based = 3

class SPEN:
  def __init__(self,config):
    self.config = config
    self.x = tf.placeholder(tf.float32, shape=[None, self.config.input_num], name="InputX")
    self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name="LearningRate")
    self.dropout_ph = tf.placeholder(tf.float32, shape=[], name="Dropout")
    self.embedding=None

  def init(self):
    init_op = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(init_op)
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

  def fnet_variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen/fx")

  def pred_variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen/pred")

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
    return np.reshape(cat, (size[0], self.config.output_num , self.config.dimension))

  def indicator_to_var(self, id):
    size = np.shape(id)
    y_cat_indicator = np.reshape(id, (size[0], self.config.output_num, self.config.dimension))
    y_m = np.argmax(y_cat_indicator, 2)
    return y_m


  def reduce_learning_rate(self,factor):
    self.config.learning_rate *= factor

  def get_energy(self, xinput=None, yinput=None, embedding=None, reuse=False):
    raise NotImplementedError

  def ssvm_training(self):
    self.margin_weight_ph = tf.placeholder(tf.float32, shape=[], name="Margin")
    self.yp = tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYP")
    self.yt = tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYT")

    self.energy_yp = self.get_energy(xinput=self.x, yinput=self.yp, embedding=self.embedding)
    self.energy_yt = self.get_energy(xinput=self.x, yinput=self.yt, embedding=self.embedding, reuse=True)

    self.ce = -tf.reduce_sum(self.yt * tf.log( tf.maximum(self.yp, 1e-20)), 1)
    self.loss_augmented_energy = self.energy_yp + self.ce * self.margin_weight_ph
    self.loss_augmented_energy_ygradient = tf.gradients(self.loss_augmented_energy, self.yp)[0]

    self.energy_ygradient = tf.gradients(self.energy_yp, self.yp)[0]

    self.objective = tf.reduce_sum( tf.maximum( self.loss_augmented_energy - self.energy_yt, 0.0)) \
                     + self.config.l2_penalty * self.get_l2_loss()

    self.num_update = tf.reduce_sum(tf.cast( self.ce * self.margin_weight_ph > self.energy_yt - self.energy_yp, tf.float32))
    self.total_energy_yt = tf.reduce_sum(self.energy_yt)
    self.total_energy_yp = tf.reduce_sum(self.energy_yp)

    self.train_step = tf.train.AdamOptimizer(self.learning_rate_ph).minimize(self.objective, var_list=self.spen_variables())


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
      raise NotImplementedError
    else:
      raise NotImplementedError




  def project_simplex_norm(self, y_ind):

    dim = self.config.dimension
    yd = np.reshape(y_ind, (-1, self.config.output_num, dim))
    eps = np.full(shape=np.shape(yd), fill_value=1e-10)
    y_min = np.min(yd, axis=2)
    y_min_all = np.reshape(np.repeat(y_min, dim), (-1, self.config.output_num, dim))
    yd_pos = yd - y_min_all
    yd_sum = np.reshape(np.repeat(np.sum(yd_pos,2),dim), (-1, self.config.output_num ,dim))
    yd_sum = yd_sum + eps
    yd_norm = np.divide(yd_pos, yd_sum)
    return np.reshape(yd_norm, (-1, self.config.output_num*dim))

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


  def inference(self, xinput=None, yinput=None, inf_iter=None, ascent=True, train=False):
    if inf_iter is None:
      inf_iter = self.config.inf_rate
    tflearn.is_training(is_training=train, session=self.sess)
    size = np.shape(xinput)[0]
    if yinput is not None:
      yt_ind = np.reshape(yinput, (-1, self.config.output_num*self.config.dimension))

    yp_ind = np.random.uniform(0, 1, (size, self.config.output_num * self.config.dimension))
    yp_ind = self.project_simplex_norm(yp_ind)
    i=0
    yp_a = []
    while i < inf_iter:
      if yinput is not None:
        feed_dict={self.x: xinput, self.yp: yp_ind, self.yt: yt_ind,
                    self.margin_weight_ph: self.config.margin_weight,
                    self.dropout_ph: self.config.dropout}
      else:
        feed_dict={self.x: xinput, self.yp: yp_ind,
                   self.margin_weight_ph: self.config.margin_weight,
                                     self.dropout_ph: self.config.dropout}

      g = self.sess.run(self.inf_gradient, feed_dict=feed_dict)
      if ascent:
        yp_ind = yp_ind + self.config.inf_rate * g
      else:
        yp_ind = yp_ind - self.config.inf_rate * g

      yp = self.softmax(np.reshape(yp_ind, (-1, self.config.output_num, self.config.dimension)), axis=2, theta=1)
      yp_a.append(yp)
      i += 1

    return np.array(yp_a)



  def soft_predict(self, xinput=None, train=False, inf_iter=None, ascent=True):
    self.inf_objective = self.energy_yp
    self.inf_gradient = self.energy_ygradient
    y_a = self.inference(xinput=xinput, inf_iter=inf_iter, train=train, ascent=ascent)
    return y_a[-1]

  def map_predict(self, xinput=None, train=False, inf_iter=None, ascent=True):
    yp = self.soft_predict(xinput=xinput, train=train, inf_iter=inf_iter, ascent=ascent)
    return np.argmax(yp, 2)


  def loss_augmented_soft_predict(self, xinput=None, yinput=None, train=False, inf_iter=None, ascent=True):
    self.inf_objective = self.loss_augmented_energy
    self.inf_gradient = self.loss_augmented_energy_ygradient
    h_a = self.inference(xinput=xinput, yinput=yinput, inf_iter=inf_iter, train=train, ascent=ascent)
    return h_a[-1]

  def loss_augmented_map_predict(self, xd, train=False, inf_iter=None, ascent=True):
    yp = self.loss_augmented_soft_predict(xd, train=train, inf_iter=inf_iter, ascent=ascent)
    return np.argmax(yp, 2)

  def train_batch(self, xbatch=None, ybatch=None, verbose=0):
    raise NotImplementedError

  def train_supervised_batch(self, xbatch, ybatch, verbose=0):
    tflearn.is_training(True, self.sess)


    yt_ind = self.var_to_indicator(ybatch)
    yt_ind = np.reshape(yt_ind, (-1, self.config.output_num*self.config.dimension))
    yp_ind = self.loss_augmented_soft_predict(xinput=xbatch, yinput=yt_ind, train=True, ascent=True)
    yp_ind = np.reshape(yp_ind, (-1, self.config.output_num*self.config.dimension))

    feeddic = {self.x:xbatch, self.yp: yp_ind, self.yt: yt_ind,
               self.learning_rate_ph:self.config.learning_rate,
               self.margin_weight_ph: self.config.margin_weight,
               self.dropout_ph: self.config.dropout}

    _, o,ce, n, en_yt, en_yhat = self.sess.run([self.train_step, self.objective, self.ce, self.num_update, self.total_energy_yt, self.total_energy_yp], feed_dict=feeddic)
    if verbose > 0:
      print (self.train_iter ,o,n, en_yt, en_yhat)
    return n
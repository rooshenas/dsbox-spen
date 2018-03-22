
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
  End2End = 4

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

  def end2end_training(self):
    self.inf_penalty_weight_ph = tf.placeholder(tf.float32, shape=[], name="InfPenalty")
    self.yt_ind= tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYT")
    self.yp_ind= tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYP")
    self.energy_y = self.get_energy(xinput=self.x, yinput=self.yp_ind, embedding=self.embedding)
    current_yp_ind = self.yp_ind
    self.objective = 0.0
    self.yp_ar = []
    for i in range(int(self.config.inf_iter)):
      next_yp_ind = current_yp_ind + self.config.inf_rate * tf.gradients(self.energy_y, self.yp_ind)[0]
      current_yp_ind = next_yp_ind
      yp_matrix = tf.reshape(current_yp_ind, [-1, self.config.output_num, self.config.dimension])
      yp_current = tf.nn.softmax(yp_matrix, 2)
      yp_ind = tf.reshape(yp_current, [-1, self.config.output_num * self.config.dimension])
      l = -tf.reduce_sum(self.yt_ind * tf.log(tf.maximum(yp_ind, 1e-20)))
      self.objective = 0.6*self.objective + 0.4*l
      self.yp_ar.append(yp_current)

    self.yp = self.yp_ar[-1] #self.get_prediction_net(input=self.h_state)
    #self.yp_ind = tf.reshape(self.yp, [-1, self.config.output_num * self.config.dimension], name="reshaped")
    #self.objective = -tf.reduce_sum(self.yt_ind * tf.log( tf.maximum(self.yp_ind, 1e-20)))
    self.train_step = self.optimizer.minimize(self.objective)



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

    self.train_step = self.optimizer.minimize(self.objective, var_list=self.spen_variables())

  def rank_based_training(self):
    self.margin_weight_ph = tf.placeholder(tf.float32, shape=[], name="Margin")
    self.value_h = tf.placeholder(tf.float32, shape=[None])
    self.value_l = tf.placeholder(tf.float32, shape=[None])
    self.yp_h_ind = tf.placeholder(tf.float32,
                          shape=[None, self.config.output_num * self.config.dimension],
                          name="YP_H")


    self.yp_l_ind = tf.placeholder(tf.float32,
                          shape=[None, self.config.output_num * self.config.dimension],
                          name="YP_L")

    self.energy_yh = self.get_energy(xinput=self.x, yinput=self.yp_h_ind, embedding=self.embedding,
                                     reuse=self.config.pretrain)
    self.energy_yl = self.get_energy(xinput=self.x, yinput=self.yp_l_ind, embedding=self.embedding,
                                     reuse=True)


    self.energy_yp = self.energy_yh
    self.yp = self.yp_h_ind

    self.energy_ygradient = tf.gradients(self.energy_yp, self.yp)[0]

    vloss = 0
    for v in self.spen_variables():
      vloss = vloss + tf.nn.l2_loss(v)

    obj1 = tf.reduce_sum( tf.maximum( (self.value_h - self.value_l)*self.margin_weight_ph - self.energy_yh + self.energy_yl, 0.0))
    self.vh_sum = tf.reduce_sum (self.value_h)
    self.vl_sum = tf.reduce_sum (self.value_l)
    self.eh_sum = tf.reduce_sum(self.energy_yh)
    self.el_sum = tf.reduce_sum(self.energy_yl)
    self.objective = obj1 +  self.config.l2_penalty * vloss #+ obj2
    self.num_update = tf.reduce_sum(tf.cast( (self.value_h - self.value_l)*self.margin_weight_ph  >= (self.energy_yh - self.energy_yl), tf.float32))
    self.train_step = self.optimizer.minimize(self.objective, var_list=self.spen_variables())
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

  def createOptimizer(self):
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)

  def construct(self, training_type = TrainingType.SSVM ):
    if training_type == TrainingType.SSVM:
      return self.ssvm_training()
    elif training_type == TrainingType.Rank_Based:
      return self.rank_based_training()
    elif training_type == TrainingType.End2End:
      return self.end2end_training()
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


  def inference(self, xinput=None, yinput=None, inf_iter=None, ascent=True, train=False):
    if inf_iter is None:
      inf_iter = self.config.inf_iter
    tflearn.is_training(is_training=train, session=self.sess)
    size = np.shape(xinput)[0]
    if yinput is not None:
      yt_ind = np.reshape(yinput, (-1, self.config.output_num*self.config.dimension))

    yp_ind = np.random.uniform(0, 1, (size, self.config.output_num * self.config.dimension))
    yp_ind = self.project_simplex_norm(yp_ind)
    i = 0
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

      yp_ind = self.project_simplex_norm(yp_ind)
      yp = np.reshape(yp_ind, (-1,self.config.output_num, self.config.dimension))
      yp_a.append(yp)
      i += 1

    return np.array(yp_a)

  def evaluate(self, xinput=None, yinput=None, yt=None):
    raise NotImplementedError



  def get_first_large_consecutive_diff(self, xinput=None, inf_iter=None, ascent=True):
    self.inf_objective = self.energy_yp
    self.inf_gradient = self.energy_ygradient

    y_a = self.inference(xinput=xinput, train=True, ascent=ascent, inf_iter=inf_iter)


    en_a = np.array([self.sess.run(self.inf_objective,
                feed_dict={self.x: xinput,
                           self.yp: np.reshape(y_i, (-1,self.config.output_num*self.config.dimension)),
                           self.dropout_ph: self.config.dropout})
                     for y_i in y_a ])
    f_a = np.array([self.evaluate(xinput=xinput, yinput=np.argmax(y_i,2)) for y_i in y_a])


    print (np.average(en_a, axis=1))
    print (np.average(f_a, axis=1))

    size = np.shape(xinput)[0]
    t = np.array(range(size))
    f1 = []
    f2 = []
    y1 = []
    y2 = []
    x = []
    k = 0
    it = np.shape(y_a)[0]
    for k in range(it-1):
      for i in t:
        if f_a[k,i] > f_a[k+1,i]:
          i_h = k
          i_l = k + 1
        else:
          i_l = k
          i_h = k + 1

        f_h = f_a[i_h,i]
        f_l = f_a[i_l,i]
        e_h = en_a[i_h,i]
        e_l = en_a[i_l,i]

        violation = (f_h - f_l)*self.config.margin_weight - e_h + e_l
        if violation > 0:
          f1.append(f_h)
          f2.append(f_l)
          y1.append((y_a[i_h,i,:]))
          y2.append((y_a[i_l,i,:]))
          x.append(xinput[i,:])

    x = np.array(x)
    f1 = np.array(f1)
    f2 = np.array(f2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    return x, y1, y2, f1, f2



  def soft_predict(self, xinput=None, train=False, inf_iter=None, ascent=True, end2end=False):
    tflearn.is_training(is_training=train, session=self.sess)
    if end2end:
      # h_init = np.random.normal(0, 1, size=(np.shape(xinput)[0], self.config.hidden_num))
      yp_ind_init = np.random.normal(0, 1, size=(np.shape(xinput)[0], self.config.output_num*self.config.dimension))
      feeddic = {self.x: xinput,
                 self.yp_ind: yp_ind_init,
                 self.inf_penalty_weight_ph: self.config.inf_penalty,
                 self.dropout_ph: self.config.dropout}
      yp = self.sess.run(self.yp, feed_dict=feeddic)
    else:
      self.inf_objective = self.energy_yp
      self.inf_gradient = self.energy_ygradient
      y_a = self.inference(xinput=xinput, inf_iter=inf_iter, train=train, ascent=ascent)
      yp =  y_a[-1]
    return yp

  def map_predict_trajectory(self, xinput=None, train=False, inf_iter=None, ascent=True, end2end=False):
    if end2end:
      tflearn.is_training(train, self.sess)
      yp_ind_init = np.random.normal(0, 1, size=(np.shape(xinput)[0], self.config.output_num*self.config.dimension))
      feeddic = {self.x: xinput,
                 self.yp_ind: yp_ind_init,
                 self.inf_penalty_weight_ph: self.config.inf_penalty,
                 self.dropout_ph: self.config.dropout}
      soft_yp_ar = self.sess.run(self.yp_ar, feed_dict=feeddic)
      yp_ar = [np.argmax(yp, 2) for yp in soft_yp_ar]
      return yp_ar
    else:
      raise NotImplementedError

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


  def train_unsupervised_batch(self, xbatch=None, verbose=0):
    tflearn.is_training(True, self.sess)
    x_b, y_h, y_l, l_h, l_l = self.get_first_large_consecutive_diff(xinput=xbatch, ascent=True)
    if np.size(l_h) > 1:
      _, o1, n1, v1, v2, e1, e2  = self.sess.run([self.train_step, self.objective, self.num_update, self.vh_sum, self.vl_sum, self.eh_sum, self.el_sum],
              feed_dict={self.x:x_b,
                         self.yp_h_ind:np.reshape(y_h, (-1, self.config.output_num * self.config.dimension)),
                         self.yp_l_ind:np.reshape(y_l, (-1, self.config.output_num * self.config.dimension)),
                         self.value_l: l_l,
                         self.value_h: l_h,
                         self.learning_rate_ph:self.config.learning_rate,
                         self.dropout_ph: self.config.dropout,
                         self.margin_weight_ph: self.config.margin_weight})
      if verbose>0:
        print (self.train_iter, o1, n1, v1,v2, e1,e2, np.shape(xbatch)[0], np.shape(x_b)[0])
    else:
      if verbose>0:
        print ("skip")
    return


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

  def train_supervised_e2e_batch(self, xbatch, ybatch, verbose=0):
    tflearn.is_training(True, self.sess)
    yt_ind = self.var_to_indicator(ybatch)
    yt_ind = np.reshape(yt_ind, (-1, self.config.output_num * self.config.dimension))
    yp_init = np.random.normal(0, 1, size=(np.shape(xbatch)[0], self.config.dimension * self.config.output_num))
    feeddic = {self.x: xbatch, self.yt_ind: yt_ind,
               self.yp_ind: yp_init,
               self.learning_rate_ph: self.config.learning_rate,
               self.inf_penalty_weight_ph: self.config.inf_penalty,
               self.dropout_ph: self.config.dropout}

    _, o = self.sess.run([self.train_step, self.objective], feed_dict=feeddic)
    if verbose > 0:
      print(self.train_iter, o)
    return o
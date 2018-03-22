
import tensorflow as tf
import numpy as np

import tflearn
import tflearn.initializations as tfi
from enum import Enum
import math


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

  def energy_g_variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen/en/en.g")


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
        cat[i, j, int(k)] = 1.0
    return np.reshape(cat, (size[0], self.config.output_num , self.config.dimension))

  def indicator_to_var(self, ind):
    size = np.shape(ind)
    y_cat_indicator = np.reshape(ind, (size[0], self.config.output_num, self.config.dimension))
    y_m = np.argmax(y_cat_indicator, 2)
    return y_m


  def reduce_learning_rate(self,factor):
    self.config.learning_rate *= factor

  def get_energy(self, xinput=None, yinput=None, embedding=None, reuse=False):
    raise NotImplementedError

  def createOptimizer(self):
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)

  def get_prediction_net(self, input=None, reuse=False):
    raise NotImplementedError

  def get_feature_net(self, xinput, output_num, embedding=None, reuse=False):
    raise NotImplementedError

  def end2end_training(self):
    self.inf_penalty_weight_ph = tf.placeholder(tf.float32, shape=[], name="InfPenalty")
    self.h = tf.placeholder(tf.float32, shape=[None, self.config.hidden_num], name="hinput")
    self.yt_ind= tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYT")
    #self.h = self.get_feature_net(self.x, self.config.hidden_num, embedding=self.embedding)

    self.h_penalty =   self.inf_penalty_weight_ph* tf.reduce_sum(tf.square(self.h),1)
    self.avg_penalty = tf.reduce_mean(self.h_penalty)
    self.energy_h = self.get_energy(xinput=self.x, yinput=self.h, embedding=self.embedding) - self.h_penalty
    h_current = self.h
    self.objective = 0.0
    self.yp_ar = [self.get_prediction_net(input=h_current)]
    for i in range(int(self.config.inf_iter)):
      h_next = h_current + self.config.inf_rate * tf.gradients(self.energy_h, self.h)[0]


      h_current = h_next
      yp_current = self.get_prediction_net(input=h_current, reuse=True)
      ind = tf.reshape(yp_current, [-1, self.config.output_num * self.config.dimension])
      l = -tf.reduce_sum(self.yt_ind * tf.log(tf.maximum(ind, 1e-20)))
      self.objective = 0.2*self.objective + 0.8*l
      self.yp_ar.append(yp_current)

    #self.opjective = l
    self.h_state = h_current
    self.yp = self.yp_ar[-1] #self.get_prediction_net(input=self.h_state)

    self.yp_ind = tf.reshape(self.yp, [-1, self.config.output_num * self.config.dimension], name="reshaped")
    #self.objective = -tf.reduce_sum(self.yt_ind * tf.log( tf.maximum(self.yp_ind, 1e-20)))
    self.train_step = self.optimizer.minimize(self.objective)


  def ssvm_training(self):
    self.margin_weight_ph = tf.placeholder(tf.float32, shape=[], name="Margin")
    self.inf_penalty_weight_ph = tf.placeholder(tf.float32, shape=[], name="InfPenalty")
    self.yp_ind = tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYP")
    self.yt_ind = tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYT")

    self.y_penalty =   self.inf_penalty_weight_ph* tf.reduce_sum(tf.square(self.yp_ind),1)
    self.yt_penalty =  self.inf_penalty_weight_ph* tf.reduce_sum(tf.square(self.yt_ind),1)


    self.energy_yp = self.get_energy(xinput=self.x, yinput=self.yp_ind, embedding=self.embedding) - self.y_penalty
    self.energy_yt = self.get_energy(xinput=self.x, yinput=self.yt_ind, embedding=self.embedding, reuse=True) - self.yt_penalty

    yp_ind_2 =  tf.reshape(self.yp_ind, [-1, self.config.output_num, self.config.dimension], name="res1")
    yp_ind_sm = tf.nn.softmax(yp_ind_2, name="sm")
    self.yp = tf.reshape(yp_ind_sm, [-1,self.config.output_num*self.config.dimension], name="res2")

    self.ce = -tf.reduce_sum(self.yt_ind * tf.log( tf.maximum(self.yp, 1e-20)), 1)
    self.en = -tf.reduce_sum(self.yp * tf.log( tf.maximum(self.yp, 1e-20)), 1)

    self.loss_augmented_energy = self.energy_yp + self.ce * self.margin_weight_ph #+ self.y_penalty
    self.loss_augmented_energy_ygradient = tf.gradients(self.loss_augmented_energy , self.yp_ind)[0]

    self.energy_ygradient = tf.gradients(self.energy_yp, self.yp_ind)[0]


    self.objective = tf.reduce_sum( tf.maximum( self.loss_augmented_energy - self.energy_yt, 0.0)) \
                     + self.config.l2_penalty * self.get_l2_loss()

    self.num_update = tf.reduce_sum(tf.cast( self.ce * self.margin_weight_ph > self.energy_yt - self.energy_yp, tf.float32))
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
    self.yp_h = tf.reshape(yp_ind_sm_h, [-1,self.config.output_num*self.config.dimension])

    yp_ind_sm_l = tf.nn.softmax(tf.reshape(self.yp_l_ind, [-1, self.config.output_num, self.config.dimension]))
    self.yp_l = tf.reshape(yp_ind_sm_l, [-1,self.config.output_num*self.config.dimension])


    self.value_h = tf.placeholder(tf.float32, shape=[None])
    self.value_l = tf.placeholder(tf.float32, shape=[None])

    #self.yh_penalty =  self.inf_penalty_weight_ph * tf.reduce_logsumexp(self.yp_h_ind ,1)
    #self.yl_penalty =  self.inf_penalty_weight_ph * tf.reduce_logsumexp(self.yp_l_ind, 1)

    self.yh_penalty =  self.inf_penalty_weight_ph * tf.maximum(tf.reduce_sum(tf.square(self.yp_h_ind), 1) , 0)
    self.yl_penalty =  self.inf_penalty_weight_ph * tf.maximum(tf.reduce_sum(tf.square(self.yp_l_ind), 1) , 0)

    self.energy_yh = self.get_energy(xinput=self.x, yinput=self.yp_h_ind, embedding=self.embedding, reuse=self.config.pretrain) - self.yh_penalty
    self.energy_yl = self.get_energy(xinput=self.x, yinput=self.yp_l_ind, embedding=self.embedding, reuse=True) - self.yl_penalty



    self.yp_ind = self.yp_h_ind
    self.yp = self.yp_h
    self.energy_yp = self.energy_yh

    #self.en = -tf.reduce_sum(self.yp * tf.log( tf.maximum(self.yp, 1e-20)), 1)

    self.energy_ygradient = tf.gradients(self.energy_yp, self.yp_ind)[0]



    self.objective = tf.reduce_mean( tf.maximum(
              (self.value_h - self.value_l)*self.margin_weight_ph - self.energy_yh + self.energy_yl, 0.0)) \
                     + self.config.l2_penalty * self.get_l2_loss()


    self.num_update = tf.reduce_sum(tf.cast(
      (self.value_h - self.value_l)*self.margin_weight_ph > (self.energy_yh - self.energy_yl), tf.float32))
    self.vh_sum = tf.reduce_sum(self.value_h)
    self.vl_sum = tf.reduce_sum(self.value_l)
    self.eh_sum = tf.reduce_sum(self.energy_yh)
    self.el_sum = tf.reduce_sum(self.energy_yl)
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


  def softmax2(self, y, theta=1.0, axis=None):
    y = self.project_simplex_norm(np.reshape(y, (-1, self.config.output_num*self.config.dimension)))
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
      inf_iter = self.config.inf_iter
    tflearn.is_training(is_training=train, session=self.sess)
    size = np.shape(xinput)[0]
    if yinput is not None:
      yt_ind = np.reshape(yinput, (-1, self.config.output_num*self.config.dimension))

    #yp_ind = np.random.uniform(0, 1, (size, self.config.output_num * self.config.dimension))

    #y = np.full((size, self.config.output_num), fill_value=5)
    y = np.random.randint(0, self.config.dimension, (size, self.config.output_num))
    yp_ind = np.reshape(self.var_to_indicator(y), (-1, self.config.output_num* self.config.dimension))
    #yp_ind = np.zeros((size, self.config.output_num * self.config.dimension))
    #yp_ind = self.project_simplex_norm(yp_ind)
    i=0
    yp_a = []
    g_m = np.zeros(np.shape(yp_ind))
    alpha = 1.0
    mean = np.zeros(shape=np.shape(yp_ind))


    #vars = self.sess.run(self.energy_variables())
    #print "W: ", np.sum(np.square(vars))
    while i < inf_iter:
      if yinput is not None:
        feed_dict={self.x: xinput, self.yp_ind: yp_ind, self.yt_ind: yt_ind,
                    self.margin_weight_ph: self.config.margin_weight,
                    self.inf_penalty_weight_ph: self.config.inf_penalty,
                    self.dropout_ph: self.config.dropout}
      else:
        feed_dict={self.x: xinput, self.yp_ind: yp_ind,
                   self.margin_weight_ph: self.config.margin_weight,
                   self.inf_penalty_weight_ph: self.config.inf_penalty,
                                     self.dropout_ph: self.config.dropout}

      g, e = self.sess.run([self.inf_gradient, self.inf_objective], feed_dict=feed_dict)
      gnorm = np.linalg.norm(g, axis=1)
      yp = self.softmax(np.reshape(yp_ind, (-1, self.config.output_num, self.config.dimension)), axis=2, theta=1)
      if self.config.loglevel > 2:
        print ("energy:", np.average(e),  "yind:", np.average(np.sum(np.square(yp_ind),1)),
             "gnorm:",np.average(gnorm), "yp:", np.average(np.max(yp, 2)))

      #g = np.clip(g,-10, 10)
      if train:
        #noise = np.random.normal(mean, inf_iter*np.abs(g) / math.sqrt((1+i)), size=np.shape(g))
        noise = np.random.normal(mean, 0.1*np.average(gnorm) / math.sqrt((1+i)), size=np.shape(g))
        #noise = np.random.normal(mean, np.abs(g), size=np.shape(g))
      else:
        noise = np.zeros(shape=np.shape(g))
        #noise = np.random.normal(mean, 100.0 / math.sqrt(1+i), size=np.shape(g))
      g_m = alpha * g + ( 1-alpha)* g_m
      if ascent:
        yp_ind = yp_ind + self.config.inf_rate * (g_m+noise)
      else:
        yp_ind = yp_ind - self.config.inf_rate * (g_m+noise)

      #yp = self.softmax(np.reshape(yp_ind, (-1, self.config.output_num, self.config.dimension)), axis=2, theta=1)
      #yp = self.softmax(np.reshape(yp_ind, (-1, self.config.output_num, self.config.dimension)), axis=2, theta=1)
      #yp_ind = self.project_simplex_norm(yp_ind)
      #yp_ind = self.softmax(np.reshape(yp_ind, (-1, self.config.output_num, self.config.dimension)), axis=2, theta=1)
      #yp_a.append(yp_ind)
      #yp_ind = np.reshape(yp_ind, (-1, self.config.output_num*self.config.dimension))
      #yp_proj = self.project_simplex_norm(yp_ind)



      #yp_ind = yp_proj
      #yp_proj = np.reshape(yp_proj, (-1, self.config.output_num, self.config.dimension))
      #yp_a.append(yp_proj)



      yp_a.append(np.reshape(yp_ind, (-1, self.config.output_num, self.config.dimension)))
      i += 1

    return np.array(yp_a)

  def evaluate(self, xinput=None, yinput=None, yt=None):
    raise NotImplementedError

  def get_first_large_consecutive_diff(self, xinput=None, yt=None, inf_iter=None, ascent=True):
    self.inf_objective = self.energy_yp
    self.inf_gradient = self.energy_ygradient

    y_a = self.inference( xinput=xinput, train=True, ascent=ascent, inf_iter=inf_iter)

    y_a = y_a[-10:]

    en_a = np.array([self.sess.run(self.inf_objective,
                feed_dict={self.x: xinput,
                           self.yp_ind: np.reshape(y_i, (-1,self.config.output_num*self.config.dimension)),
                           self.inf_penalty_weight_ph: self.config.inf_penalty,
                           self.dropout_ph: self.config.dropout})
                     for y_i in y_a ])
    f_a = np.array([self.evaluate(xinput=xinput, yinput=np.argmax(y_i,2), yt=yt) for y_i in y_a])


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

  def get_all_diff(self, xinput=None, yinput=None, inf_iter=None, ascent=True):
    self.inf_objective = self.energy_yp
    self.inf_gradient = self.energy_ygradient
    y_a = self.inference(xinput=xinput, inf_iter=inf_iter, train=True, ascent=ascent)
    yp_a = np.array([self.softmax(yp) for yp in y_a])

    en_a = np.array([self.sess.run(self.inf_objective,
                feed_dict={self.x: xinput,
                           self.yp_ind: np.reshape(y_i, (-1,self.config.output_num*self.config.dimension)),
                           self.inf_penalty_weight_ph: self.config.inf_penalty,
                           self.dropout_ph: self.config.dropout})
                     for y_i in y_a ])

    ce_a = np.array([np.sum(yinput * np.log(1e-20 + np.reshape(y_p, (-1, self.config.output_num * self.config.dimension))),1) for y_p in yp_a])
    #f_a = np.array([self.evaluate(xinput=xinput, yinput=np.argmax(y_i,2), yt=np.argmax(np.reshape(yinput, (-1, self.config.output_num, self.config.dimension)),2)) for y_i in y_a])

    e_t = self.sess.run(self.inf_objective,
                                   feed_dict={self.x: xinput,
                                              self.yp_ind: np.reshape(yinput, (
                                              -1, self.config.output_num * self.config.dimension)),
                                              self.inf_penalty_weight_ph: self.config.inf_penalty,
                                              self.dropout_ph: self.config.dropout})
    print (np.average(en_a, axis=1))
    print (np.average(ce_a, axis=1))

    size = np.shape(xinput)[0]
    t = np.array(range(size))
    y = []
    yp = []
    x = []
    it = np.shape(y_a)[0]
    for k in range(it):
      for i in t:

        violation = (-ce_a[k,i]) * self.config.margin_weight - e_t[i] + en_a[k,i]
        print (e_t[i], en_a[k,i], ce_a[k,i], violation)
        if violation > 0:
          yp.append((y_a[k,i,:]))
          x.append(xinput[i,:])
          y.append(yinput[i,:])
    x = np.array(x)
    y = np.array(y)
    yp = np.array(yp)

    return x, y, yp

  def h_predict(self, xinput=None, train=False, inf_iter=None, ascent=True):
    tflearn.is_training(is_training=train, session=self.sess)
    h_init = np.random.normal(0, 1, size=(np.shape(xinput)[0], self.config.hidden_num))
    feeddic = {self.x: xinput,
                 self.h: h_init,
                 self.inf_penalty_weight_ph: self.config.inf_penalty,
                 self.dropout_ph: self.config.dropout}
    h = self.sess.run(self.h_state ,feed_dict=feeddic)
    return h


  def soft_predict(self, xinput=None, train=False, inf_iter=None, ascent=True, end2end=False):
    tflearn.is_training(is_training=train, session=self.sess)
    if end2end:
      #h_init = np.random.normal(0, 1, size=(np.shape(xinput)[0], self.config.hidden_num))
      h_init = np.random.normal(0, 1, size=(np.shape(xinput)[0], self.config.hidden_num))
      feeddic = {self.x: xinput,
                 self.h: h_init,
                 self.inf_penalty_weight_ph: self.config.inf_penalty,
                 self.dropout_ph: self.config.dropout}
      yp = self.sess.run(self.yp, feed_dict=feeddic)
    else:

      self.inf_objective = self.energy_yp
      self.inf_gradient = self.energy_ygradient
      y_a = self.inference(xinput=xinput, inf_iter=inf_iter, train=train, ascent=ascent)

      en_a = np.array([self.sess.run(self.energy_yp,
                        feed_dict={self.x: xinput,
                                   self.yp_ind: np.reshape(y_i, (-1, self.config.output_num * self.config.dimension)),
                                   self.inf_penalty_weight_ph: self.config.inf_penalty,
                                   self.dropout_ph: self.config.dropout}) for y_i in y_a])
      try:
        for i in range(len(en_a)):
          y_i = y_a[i]
          f_i = np.array(self.evaluate(xinput=xinput, yinput=np.argmax(y_i, 2)))
          print ("----------------------------")
          print (i, np.average(en_a[i,:]), np.average(f_i))


        y_ans = y_a[-1]

        #f_a = np.array([self.evaluate(xinput=xinput, yinput=np.argmax(y_i,2)) for y_i in y_a])
        #ind = np.argmax(f_a,0) if ascent else np.argmin(f_a, 0)
        #y_ans = np.array([y_a[ind[i],i,:] for i in range(np.shape(xinput)[0])])
      except:
        y_ans = y_a[-1]
      yp = self.softmax(y_ans, axis=2, theta=1)
      en = -np.sum(yp * np.log(yp+1e-20))


      print (np.average(np.square(yp)), en, np.average(np.max(yp,2)))

    return yp

  def map_predict_trajectory(self, xinput=None, train=False, inf_iter=None, ascent=True, end2end=False):
    if end2end:
      tflearn.is_training(train, self.sess)
      h_init = np.random.normal(0, 1, size=(np.shape(xinput)[0], self.config.hidden_num))
      feeddic = {self.x: xinput,
                 self.h: h_init,
                 self.inf_penalty_weight_ph: self.config.inf_penalty,
                 self.dropout_ph: self.config.dropout}
      soft_yp_ar = self.sess.run(self.yp_ar, feed_dict=feeddic)
      yp_ar =  [np.argmax(yp, 2) for yp in soft_yp_ar]
      return yp_ar
    else:
      raise NotImplementedError


  def map_predict(self, xinput=None, train=False, inf_iter=None, ascent=True, end2end=False):
    yp = self.soft_predict(xinput=xinput, train=train, inf_iter=inf_iter, ascent=ascent, end2end=end2end)
    return np.argmax(yp, 2)

  #def inference_trajectory(self):

  def loss_augmented_soft_predict(self, xinput=None, yinput=None, train=False, inf_iter=None, ascent=True):
    self.inf_objective = self.loss_augmented_energy
    self.inf_gradient = self.loss_augmented_energy_ygradient
    h_a = self.inference(xinput=xinput, yinput=yinput, inf_iter=inf_iter, train=train, ascent=ascent)
    #
    # en_a  = np.array([self.sess.run(self.inf_objective,
    #               feed_dict={self.x: xinput,
    #                          self.yp_ind: np.reshape(ind_i, (-1, self.config.output_num * self.config.dimension)),
    #                          self.yt: yinput,
    #                          self.margin_weight_ph: self.config.margin_weight,
    #                         self.inf_penalty_weight_ph: self.config.inf_penalty,
    #                         self.dropout_ph: self.config.dropout}) for ind_i in h_a])
    #
    # print ("en:", en_a[:,0])


    return self.softmax(h_a[-1], axis=2, theta=1)

  def get_adverserial_predict(self, xinput=None, yinput=None, train=False, inf_iter=None, ascent=True):
    self.inf_objective = self.energy_yp
    self.inf_gradient = self.energy_ygradient
    yp_a = self.inference(xinput=xinput, yinput=yinput, inf_iter=inf_iter, train=train, ascent=ascent)
    yp_a = np.array([self.softmax(yp) for yp in yp_a])
    en_a  = np.array([self.sess.run(self.inf_objective,
                  feed_dict={self.x: xinput,
                             self.yp_ind: np.reshape(ind_i, (-1, self.config.output_num * self.config.dimension)),
                             self.yt_ind: yinput,
                             self.margin_weight_ph: self.config.margin_weight,
                             self.inf_penalty_weight_ph: self.config.inf_penalty,
                            self.dropout_ph: self.config.dropout}) for ind_i in yp_a])

    ce_a = np.array([-np.sum(yinput * np.log(1e-20 + np.reshape(y_p, (-1, self.config.output_num * self.config.dimension))),1) for y_p in yp_a])
    print ("en:", np.average(en_a, axis=1), "ce:", np.average(ce_a, axis=1))

    return self.softmax(yp_a[-1], axis=2, theta=1)

  def loss_augmented_map_predict(self, xd, train=False, inf_iter=None, ascent=True):
    yp = self.loss_augmented_soft_predict(xd, train=train, inf_iter=inf_iter, ascent=ascent)
    return np.argmax(yp, 2)

  def train_batch(self, xbatch=None, ybatch=None, verbose=0):
    raise NotImplementedError


  def train_unsupervised_batch(self, xbatch=None, ybatch=None, verbose=0):
    tflearn.is_training(True, self.sess)
    x_b, y_h, y_l, l_h, l_l = self.get_first_large_consecutive_diff(xinput=xbatch, yt=ybatch, ascent=True)
    if np.size(l_h) > 1:
      _, o1, n1, v1, v2, e1, e2  = self.sess.run([self.train_step, self.objective, self.num_update, self.vh_sum, self.vl_sum, self.eh_sum, self.el_sum],
              feed_dict={self.x:x_b,
                         self.yp_h_ind:np.reshape(y_h, (-1, self.config.output_num * self.config.dimension)),
                         self.yp_l_ind:np.reshape(y_l, (-1, self.config.output_num * self.config.dimension)),
                         self.value_l: l_l,
                         self.value_h: l_h,
                         self.learning_rate_ph:self.config.learning_rate,
                         self.dropout_ph: self.config.dropout,
                         self.inf_penalty_weight_ph: self.config.inf_penalty,
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
    #xd, yd, yp_ind = self.get_all_diff(xinput=xbatch, yinput=yt_ind, ascent=True, inf_iter=10)
    yp_ind = self.loss_augmented_soft_predict(xinput=xbatch, yinput=yt_ind, train=True, ascent=True)
    yp_ind = np.reshape(yp_ind, (-1, self.config.output_num*self.config.dimension))
    #yt_ind = np.reshape(yd, (-1, self.config.output_num*self.config.dimension))

    feeddic = {self.x:xbatch, self.yp_ind: yp_ind, self.yt_ind: yt_ind,
               self.learning_rate_ph:self.config.learning_rate,
               self.margin_weight_ph: self.config.margin_weight,
               self.inf_penalty_weight_ph: self.config.inf_penalty,
               self.dropout_ph: self.config.dropout}

    _, o,ce, n, en_yt, en_yhat = self.sess.run([self.train_step, self.objective, self.ce, self.num_update, self.total_energy_yt, self.total_energy_yp], feed_dict=feeddic)
    if verbose > 0:
      print (self.train_iter ,o,n, en_yt, en_yhat)
    return n

  def train_supervised_e2e_batch(self, xbatch, ybatch, verbose=0):
    tflearn.is_training(True, self.sess)
    yt_ind = self.var_to_indicator(ybatch)
    yt_ind = np.reshape(yt_ind, (-1, self.config.output_num*self.config.dimension))
    h_init = np.random.normal(0, 1, size=(np.shape(xbatch)[0], self.config.hidden_num))
    feeddic = {self.x:xbatch, self.yt_ind: yt_ind,
               self.h: h_init,
               self.learning_rate_ph:self.config.learning_rate,
               self.inf_penalty_weight_ph: self.config.inf_penalty,
               self.dropout_ph: self.config.dropout}

    _, o,p  = self.sess.run([self.train_step, self.objective, self.avg_penalty], feed_dict=feeddic)
    if verbose > 0:
      print (self.train_iter ,o, p)
    return o
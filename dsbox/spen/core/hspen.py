
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
    self.is_training = tf.placeholder(tf.float32, shape=[], name="IsTraining")
    self.dropout_ph = tf.placeholder(tf.float32, shape=[], name="Dropout")
    self.bs_ph = tf.placeholder(tf.int32, shape=[], name="BatchSize")

    self.embedding=None
    self.train_iter = 0.0


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

  def get_prediction_net(self, input=None, xinput=None, reuse=False):
    raise NotImplementedError

  def get_feature_net(self, xinput, output_num, embedding=None, reuse=False):
    raise NotImplementedError

  def get_loss(self, yt, yp):
    raise NotImplementedError

  def f1_loss(self, yt, yp):
    l = -tf.reduce_sum((tf.reduce_sum(yt * tf.log(tf.maximum(yp, 1e-20)), 1)
                        + 0.1*tf.reduce_sum((1. - yt) * tf.log(tf.maximum(1. - yp, 1e-20)), 1)))
    yp = tf.reshape(yp, [-1, self.config.output_num, self.config.dimension])
    yt = tf.reshape(yt, [-1, self.config.output_num, self.config.dimension])
    yp_ones = yp[:, :, 1]
    yt_ones = yt[:, :, 1]
    intersect = tf.reduce_sum(tf.minimum(yt_ones, yp_ones),1)
    return -tf.reduce_sum(2*intersect / (tf.reduce_sum(yt_ones,1) +tf.reduce_sum(yp_ones,1) )) + l - 1.2*tf.reduce_sum(yp_ones)

  def ce_loss(self, yt, yp):
    eps = 1e-20
    #ypc = tf.reshape(yp, (-1, self.config.output_num, self.config.dimension))
    #yp = tf.clip_by_value(yp, clip_value_min=eps, clip_value_max=1.0 - eps )
    #ytc = tf.reshape(yt, (-1, self.config.output_num, self.config.dimension))
    #l =  tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=yt, logits=yp))
    if self.config.dimension > 1:
      yp = tf.reshape(yp, (-1, self.config.output_num*self.config.dimension))
    l = -tf.reduce_sum(yt * tf.log(tf.maximum(yp, eps))) - tf.reduce_sum((1. - yt) * tf.log(tf.maximum(1. - yp , eps)))
    #if self.config.dimension == 1:
    #  self.syp = tf.reduce_mean(yp)
    #  self.syp_t = tf.reduce_mean(yp)
    #  self.pred = yp
    #  self.gt = yt
    #else:
    #  self.syp = tf.reduce_mean(tf.reduce_max(ypc,-1))
    #  self.syp_t = tf.reduce_mean(tf.reduce_max(tf.multiply(ypc, ytc), -1))

    #  self.pred = tf.argmax(ypc,axis=2)
    #  self.gt = tf.argmax(ytc, axis=2)

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

  def ce_sym_loss_b(self, yt, yp):
    l = -tf.reduce_sum((tf.reduce_sum(yt * tf.log(tf.maximum(yp, 1e-20)), 1) \
                        + 0.0001*tf.reduce_sum((1. - yt) * tf.log(tf.maximum(1. - yp, 1e-20)), 1)))
    return l

  def ce_en_loss(self, yt, yp):
    eps = 1e-30
    ypc = tf.reshape(yp, (-1, self.config.output_num, self.config.dimension))
    #yp = tf.clip_by_value(yp, clip_value_min=eps, clip_value_max=1.0 - eps )
    ytc = tf.reshape(yt, (-1, self.config.output_num, self.config.dimension))
    #l =  tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=yt, logits=yp))
    l = -tf.reduce_sum(ytc * tf.log(tf.maximum(ypc, eps))) - 0.1*tf.reduce_sum(ypc * tf.log(tf.maximum(ypc, eps)))
          #            -tf.reduce_sum((1. - ytc) * tf.log(tf.maximum(1. - ypc , eps)))
    self.syp = tf.reduce_mean(tf.reduce_max(ypc,-1))
    self.syp_t = tf.reduce_mean(tf.reduce_max(tf.multiply(ypc, ytc), -1))

    self.pred = tf.argmax(ypc,axis=2)
    self.gt = tf.argmax(ytc, axis=2)

    return l



  def mse_loss(self, yt, yp):
    #yt_max = tf.cast(tf.argmax(tf.reshape(yt, (-1, self.config.output_num, self.config.dimension)), axis=2), tf.float32)
    #yp_max = tf.cast(tf.argmax(tf.reshape(yp, (-1, self.config.output_num, self.config.dimension)), axis=2), tf.float32)
    self.syp = tf.reduce_mean(yp)
    self.syp_t = tf.reduce_mean(yt)
    self.pred = yp
    self.gt = yt
    eps = 1e-20

    l = tf.reduce_mean( tf.square((yt-yp)*255.0))
    return l

  def biased_loss(self, yt, yp):
    l = -tf.reduce_sum ((tf.reduce_sum(yt * tf.log(tf.maximum(yp, 1e-20)), 1) \
       + tf.reduce_sum((1. - yt) * tf.log(tf.maximum(1. - yp , 1e-20)), 1)) )
    yp = tf.reshape(yp, [-1, self.config.output_num, self.config.dimension])
    yp_zeros = yp[:, :, 0]
    yp_ones = yp[:,:, 1]
    en = tf.reduce_sum(yp_ones * tf.log(yp_ones), 1)
    return l + 1.2*(tf.reduce_sum(yp_zeros) - tf.reduce_sum(yp_ones)) +  0.0*tf.reduce_sum(en)

  def biased_horses_loss(self, yt, yp):
    l = -tf.reduce_sum ((tf.reduce_sum(yt * tf.log(tf.maximum(yp, 1e-20)), 1) \
       + tf.reduce_sum((1. - yt) * tf.log(tf.maximum(1. - yp , 1e-20)), 1)) )
    intersect = tf.reduce_sum(tf.multiply(yt, yp))
    union = tf.reduce_sum(yt) + tf.reduce_sum(yp) - intersect
    #intersect = tf.reduce_sum(tf.minimum(yt, yp))
    #union = tf.reduce_sum(tf.maximum(yt, yp))  # + 1e-20
    return l + union - intersect

  def get_initialization_net(self, xinput, output_size, embedding=None, reuse=False):
    raise NotImplementedError



  def iou_loss2(self,yt, yp):
    intersection = tf.reduce_sum(tf.minimum(yp,yt),1)
    union = tf.reduce_sum(tf.maximum(yp, yt),1)
    iou = tf.divide(intersection, union)
    loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), iou)
    return 100*loss

  def iou_loss(self, yt, yp):
    # code from here
    # http://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html
    # logits = tf.reshape(yp, [-1])
    # trn_labels = tf.reshape(yt, [-1])
    logits = yt
    trn_labels = yp
    '''
    Eq. (1) The intersection part - tf.mul is element-wise, 
    if logits were also binary then tf.reduce_sum would be like a bitcount here.
    '''
    inter = tf.reduce_sum(tf.multiply(logits, trn_labels), 1)

    '''
    Eq. (2) The union part - element-wise sum and multiplication, then vector sum
    '''
    union = tf.reduce_sum(tf.subtract(tf.add(logits, trn_labels), tf.multiply(logits, trn_labels)), 1)

    # Eq. (4)
    loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.div(inter, union))

    return 1000*loss  # tf.div(inter, union)

  def end2end_training_o(self):
    def body(h, g, iter, obj, yp):
      penalty_current = self.inf_penalty_weight_ph* tf.reduce_sum(tf.square(h),1)
      energy_current = self.get_energy(xinput=self.features, yinput=h, embedding=None, reuse=True) - penalty_current
      g = tf.gradients(energy_current, h)[0]
      g = tf.clip_by_value(g, clip_value_min=-0.1, clip_value_max=0.1)
      h_next = h_current  + self.config.inf_rate * g
      yp_current = self.get_prediction_net(input=h_current, xinput=self.x, reuse=True)
      ind = tf.reshape(yp_current, [-1, self.config.output_num * self.config.dimension])
      l = self.get_loss(self.yt_ind, ind)
      obj += (self.config.alpha / (self.config.inf_iter - iter + 1.0)) * l
      return [h_next, g, tf.add(iter,1.0), obj,yp_current]

    def cond(h,g, iter, obj, yp):
      #return tf.less(iter, tf.constant(5.0))
      return tf.reduce_any(tf.logical_and(tf.greater(tf.norm(g), tf.constant(0.1)),tf.less(iter, 10.0)) )



    self.inf_penalty_weight_ph = tf.placeholder(tf.float32, shape=[], name="InfPenalty")
    self.h = tf.placeholder(tf.float32, shape=[None, self.config.hidden_num], name="hinput")
    noinit = False
    try:
      h_start, features = self.get_initialization_net(self.x, self.config.hidden_num, embedding=self.embedding)
    except:
      h_start = self.h
      print ("no init network")
      noinit = True
      #raise  NotImplementedError("Should have used init model")
    self.features = features
    self.yp_h = self.get_prediction_net(input=h_start, xinput=self.x)
    self.energy_h = self.get_energy(xinput=features, yinput=h_start, embedding=None)
    self.yp_hpredict = self.get_prediction_net(input=self.h, xinput=self.x, reuse=True)
    self.yt_ind = tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYT")

    self.h_penalty =  self.inf_penalty_weight_ph * tf.reduce_sum(tf.square(self.h - h_start),1)
    self.objective = 0.0


    h_current = h_start #+ self.h
    self.objective = 0.0 #self.get_loss(self.yt_ind, self.yp_h)
    h, g, iter, obj, yp_l = tf.while_loop(cond, body, [h_start, h_start, 0.0, 0.0, self.yp_h])
    self.iter = iter
    self.g_ar  = [tf.norm(g)]
    self.objective = obj + self.config.l2_penalty * self.get_l2_loss()
    self.yp_ar = [ yp_l]
    self.h_ar = [tf.norm(h)]
    if self.config.dimension > 1:
      self.yp = tf.reshape(self.yp_ar[-1], [-1, self.config.output_num, self.config.dimension])
    else:
      self.yp = self.yp_ar[-1]

    self.train_all_step = self.optimizer.minimize(self.objective)
    self.train_step = self.optimizer.minimize(self.objective, var_list=self.spen_variables())



  def end2end_training(self):
    self.inf_penalty_weight_ph = tf.placeholder(tf.float32, shape=[], name="InfPenalty")
    self.h = tf.placeholder(tf.float32, shape=[None, self.config.hidden_num], name="hinput")
    noinit = False
    try:
      h_start, features = self.get_initialization_net(self.x, self.config.hidden_num, embedding=self.embedding)
    except:
      h_start = self.h
      print ("no init network")
      noinit = True
      #raise  NotImplementedError("Should have used init model")


    #self.yp_h  = self.get_prediction_net(input=tf.concat((h_start, tf.zeros(h_start.get_shape())), axis=1), xinput=self.x)
    self.yp_h = self.get_prediction_net(input=(h_start), xinput=self.x)
    if self.config.dimension == 1:
      self.yp_hpredict = tf.nn.sigmoid(self.get_prediction_net(input=self.h, xinput=self.x, reuse=True))
    else:
      logits = self.get_prediction_net(input=self.h, xinput=self.x, reuse=True)
      logits = tf.reshape(logits, (-1, self.config.output_num, self.config.dimension))
      self.yp_hpredict = tf.nn.softmax(logits)

    self.yt_ind = tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYT")
    #self.h = self.get_feature_net(self.x, self.config.hidden_num, embedding=self.embedding)

    self.h_penalty =  self.inf_penalty_weight_ph * tf.reduce_sum(tf.square(self.h - h_start),1)
    #self.avg_h = tf.reduce_mean(tf.square(h_start))
    #self.energy_h = self.get_energy(xinput=h_start, yinput=self.h, embedding=self.embedding) - self.h_penalty
    #self.energy_hgradient = tf.gradients(self.energy_h, self.h)[0]


    self.objective = 0.0


    self.yp_ar = [self.yp_h]
    self.en_ar = []
    self.g_ar = []
    self.pen_ar = []
    self.l_ar = []
    h_total = (1.0 / (self.config.inf_iter+1) ) * h_start
    k = (1.0 / (self.config.inf_iter+1) )
    #hnoise = tf.random_normal(shape=(self.bs_ph, self.config.hidden_num), mean=0.0, stddev=tf.norm(h_start, 1))
    h_current = h_start# + hnoise

    self.h_ar = [h_current]
    self.objective = 0.0 #self.get_loss(self.yt_ind, self.yp_h)]]
    en_total = 0.0
    for i in range(int(self.config.inf_iter)):
      #tf.square(tf.norm(h_current, axis=1) - 1)
      penalty_current = self.inf_penalty_weight_ph* tf.reduce_sum(tf.square(h_current),1)
      energy_current = self.get_energy(xinput=self.x if noinit else features, yinput=h_current, embedding=None, reuse=True if i > 0 else False) - penalty_current
      g = tf.gradients(energy_current, h_current)[0]
      g = tf.clip_by_value(g, clip_value_min=-1.0, clip_value_max=1.0)
      en_total += energy_current
      self.en_ar.append(energy_current)
      self.g_ar.append(tf.reduce_mean(tf.norm(g,1)))

      #self.pen_ar.append(penalty_current)
      #noise = tf.random_normal(shape=tf.shape(g),stddev=self.config.noise_rate*tf.norm(g)/tf.sqrt(tf.cast(i, tf.float32) + 1.0))
      noise = tf.random_normal(shape=tf.shape(g),stddev=self.config.noise_rate*tf.norm(g))

      h_next = h_current  + (self.config.inf_rate)* tf.cond(self.is_training > 0.0, lambda: g+noise, lambda: g)
      #h_next = h_current  + self.config.inf_rate * (self.config.inf_rate/tf.sqrt(tf.cast(i, tf.float32) + 1.0)) * tf.cond(self.is_training > 0.0, lambda: g+noise, lambda: g)
      h_current = h_next #+ 0.001*tf.random_normal(shape=(self.bs_ph, self.config.hidden_num), mean=0.0, stddev=tf.norm(h_current,1))
      #h_total += h_current
      #h_extend = tf.concat ((h_current, h_start), axis=1)
      #h_normal = tf.nn.softmax(h_current)
      yp_current_logits  = self.get_prediction_net(input=h_current, xinput=self.x, reuse=True)

      if self.config.dimension == 1:
        yp_current = tf.nn.sigmoid(yp_current_logits)
        #l = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=yp_current_logits, labels=self.yt_ind))
        eps=1e-10
        #yp_current = yp_current_logits
        #yp_current = tf.clip_by_value(yp_current_logits, clip_value_min=eps, clip_value_max=1-eps)

      else:
        yp_current_logits = tf.reshape(yp_current_logits, (-1, self.config.output_num, self.config.dimension))
        yp_current = tf.nn.softmax(yp_current_logits, dim=2)
        #l = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
        #    logits=yp_current_logits,
        #    labels=tf.reshape(self.yt_ind, (-1, self.config.output_num, self.config.dimension))))

      ind = tf.reshape(yp_current, [-1, self.config.output_num * self.config.dimension])
      l  = self.get_loss(self.yt_ind, ind)
      #l = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=yp_current_logits, labels=self.yt_ind))
      self.l_ar.append(tf.reduce_mean(l))
     # self.h_ar.append(h_normal)
      self.h_ar.append(h_current)
      #yp_total += yp_current
      #self.objective = (1.0 - self.config.alpha) * self.objective + self.config.alpha * l
      #self.objective += (self.config.alpha / (self.config.inf_iter - i)) * l
      #h_total += (self.config.alpha / (self.config.inf_iter - i + 1.0)) * h_current
      #k += (self.config.alpha / (self.config.inf_iter - i + 1.0))
      self.yp_ar.append(yp_current)

    #ind = tf.reshape(self.yp_ar[-1], [-1, self.config.output_num * self.config.dimension])
    #l  = self.get_loss(self.yt_ind, ind)
    #self.objective = self.get_loss(self.yt_ind, )
    #ind = tf.reshape(self.yp_ar[-1], [-1, self.config.output_num * self.config.dimension])
    #l = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=yp_current_logits, labels=self.yt_ind))
    #self.objective = self.get_loss(self.yt_ind, ind) + self.config.l2_penalty * self.get_l2_loss()
    self.objective = tf.reduce_sum(l) + self.config.l2_penalty * self.get_l2_loss() + self.ce_loss(self.yt_ind, ind)

    self.yp = self.yp_ar[-1]
    #if self.config.dimension > 1:
    #  self.yp = tf.reshape(self.yp_ar[-1], [-1, self.config.output_num, self.config.dimension])
    #else:
    #  self.yp = self.yp_ar[-1]

      #self.yp = self.get_prediction_net(input=(h_total / k), xinput=self.x, reuse=True)
      #self.yp = self.get_prediction_net(h_total / self.config.inf_it)
      #yp_ar = np.array(self.yp_ar)
      #self.yp = tf.reduce_mean(self.yp_ar, axis=-1)
    #self.get_prediction_net(input=self.h_state)


    #self.yp_mlp = self.get_prediction_net(input=h_start, xinput=self.x, reuse=True)
    #ind_mlp = tf.reshape(self.yp_mlp, [-1, self.config.output_num * self.config.dimension])
    #self.obj_mlp = self.get_loss(self.yt_ind, ind_mlp)

    #self.yp_ind = tf.reshape(self.yp, [-1, self.config.output_num * self.config.dimension], name="reshaped")
    #self.objective = -tf.reduce_sum(self.yt_ind * tf.log( tf.maximum(self.yp_ind, 1e-20)))
    self.train_all_step = self.optimizer.minimize(self.objective)
    self.train_step = self.optimizer.minimize(self.objective, var_list=self.spen_variables())

    #init_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen/init")
    #fc_var =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="pred")
    #spen_e_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen/fx")
    #spen_fx_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="spen/en")

    #self.train_pre_step = self.optimizer.minimize(self.obj_mlp, var_list=(init_var+fc_var))
    #self.train_post_step = self.optimizer.minimize(self.objective, var_list=(spen_e_var+spen_fx_var+init_var))

    #self.train_pred_step = self.optimizer.minimize(self.objective, var_list=self.pred_variables())

  # def end2end_training2(self):
  #   self.inf_penalty_weight_ph = tf.placeholder(tf.float32, shape=[], name="InfPenalty")
  #   self.h = tf.placeholder(tf.float32, shape=[None, self.config.hidden_num], name="hinput")
  #   try:
  #     h_start = self.get_initialization_net(self.x, self.config.hidden_num)
  #   except:
  #     raise NotImplementedError("Should have used init model")
  #
  #   # self.yp_h  = self.get_prediction_net(input=tf.concat((h_start, tf.zeros(h_start.get_shape())), axis=1), xinput=self.x)
  #   self.yp_h = self.get_prediction_net(input=h_start, xinput=self.x)
  #
  #   self.yt_ind = tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension],
  #                                name="OutputYT")
  #   # self.h = self.get_feature_net(self.x, self.config.hidden_num, embedding=self.embedding)
  #
  #   self.h_penalty = self.inf_penalty_weight_ph * tf.reduce_sum(tf.square(h_start), 1)
  #   self.avg_h = tf.reduce_mean(tf.square(h_start))
  #   # self.energy_h = self.get_energy(xinput=self.x, yinput=self.h, embedding=self.embedding) - self.h_penalty
  #   h_current = h_start
  #   self.objective = 0.0
  #   self.h_ar = [h_start]
  #
  #   self.yp_ar = [self.yp_h]
  #   self.en_ar = []
  #   self.g_ar = []
  #   self.pen_ar = []
  #   h_total = h_start + 0.0
  #   k = 0.0
  #   self.objective = self.get_loss(self.yt_ind, self.yp_h)
  #   for i in range(int(self.config.inf_iter)):
  #     penalty_current = self.inf_penalty_weight_ph * tf.reduce_sum(tf.square(h_current-h_start), 1)
  #     energy_current = self.get_energy(xinput=h_start, yinput=h_current, embedding=None,
  #                                      reuse=False if i == 0 else True) - penalty_current
  #     g = tf.gradients(energy_current, h_current)[0]
  #     self.en_ar.append(energy_current)
  #     self.g_ar.append(g)
  #
  #     # self.pen_ar.append(penalty_current)
  #     noise = tf.random_normal(shape=tf.shape(g),
  #                              stddev=self.config.noise_rate * tf.norm(g) / tf.sqrt(tf.cast(i, tf.float32) + 1.0))
  #
  #     h_next = h_current + self.config.inf_rate * (
  #     self.config.inf_rate / tf.sqrt(tf.cast(i, tf.float32) + 1.0)) * tf.cond(self.is_training > 0.0, lambda: g + noise,
  #                                                                             lambda: g)
  #     h_current = h_next
  #     #if k>10:
  #     h_total += h_current
  #     k += 1.0
  #     # h_extend = tf.concat ((h_current, h_start), axis=1)
  #     #yp_current = self.get_prediction_net(input=h_current, xinput=self.x, reuse=True)
  #
  #     self.h_ar.append(h_current)
  #     # self.objective = (1.0 - self.config.alpha) * self.objective + self.config.alpha * l
  #     #self.objective += (self.config.alpha / (self.config.inf_iter - i + 1.0)) * l
  #     #self.yp_ar.append(yp_current)
  #     #yp_current = self.yp = self.get_prediction_net(input=(h_total/(self.config.inf_iter+0.0)), xinput=self.x, reuse=True)
  #
  #
  #   self.yp = self.get_prediction_net(input=(h_total/k), xinput=self.x, reuse=True)
  #   ind = tf.reshape(self.yp, [-1, self.config.output_num * self.config.dimension])
  #   l = self.get_loss(self.yt_ind, ind)
  #
  #   self.objective = l + self.config.l2_penalty * self.get_l2_loss()
  #   #if self.config.dimension > 1:
  #   #  self.yp = tf.reshape(self.yp_ar[-1], [-1, self.config.output_num, self.config.dimension])
  #   #else:
  #   #  self.yp = self.yp_ar[-1] # self.get_prediction_net(h_total / self.config.inf_it)
  #   # self.get_prediction_net(input=self.h_state)
  #
  #   # self.yp_ind = tf.reshape(self.yp, [-1, self.config.output_num * self.config.dimension], name="reshaped")
  #   # self.objective = -tf.reduce_sum(self.yt_ind * tf.log( tf.maximum(self.yp_ind, 1e-20)))
  #   self.train_all_step = self.optimizer.minimize(self.objective)
  #   self.train_step = self.optimizer.minimize(self.objective, var_list=self.spen_variables())
  #
  #   #self.yp_mlp = self.get_prediction_net(input=self.h0, xinput=feature, reuse=True)
  #   # self.objective_mlp = self.get_loss(self.yt_ind, self.yp_mlp) + self.config.l2_penalty * self.get_l2_loss()
  #   self.train_pred_step = self.optimizer.minimize(self.objective, var_list=self.pred_variables())
  #   # predxh = self.predh_variables() + self.predx_variables()
  #   # hspen = self.predh_variables() + self.spen_variables()
  #   # self.train_hspen_step = self.optimizer.minimize(self.objective, var_list=hspen)
  #   # self.train_predhx_step = self.optimizer.minimize(self.objective, var_list=predxh)

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
    #tf.reset_default_graph()
    if training_type == TrainingType.SSVM:
      return self.ssvm_training()
    elif training_type == TrainingType.Rank_Based:
      return self.rank_based_training()
    elif training_type == TrainingType.End2End:
      return self.end2end_training()
    elif training_type == TrainingType.Value_Matching:
      return self.value_training()
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


  def inference(self, xinput=None, hstart=None, inf_iter=None, ascent=True, train=False):
    if inf_iter is None:
      inf_iter = self.config.inf_iter
    tflearn.is_training(is_training=train, session=self.sess)
    h = hstart
    i=0
    h_a = []
    en_ar = []
    while i < inf_iter:
      feed_dict={self.x: xinput, self.h: h,
                 self.inf_penalty_weight_ph: self.config.inf_penalty,
                 self.dropout_ph: self.config.dropout}
      g, e = self.sess.run([self.inf_gradient, self.inf_objective], feed_dict=feed_dict)
      if ascent:
        h = h + self.config.inf_rate * (g)
      else:
        h = h - self.config.inf_rate * (g)
      h_a.append(h)
      i += 1
      if self.config.loglevel > 3:
        #print (np.shape(g), np.shape(e))
        print (np.average(e))
        #print (np.average(np.linalg.norm(g, axis=1)), np.average(np.linalg.norm(h, axis=1), np.average(e)))
    return np.array(h_a)

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
                 self.is_training : 1.0 if train else 0.0,
                  self.bs_ph: np.shape(xinput)[0],
                 self.dropout_ph: self.config.dropout}
    h_ar = self.sess.run(self.h_ar ,feed_dict=feeddic)
    return h_ar[-1]


  def h_trajectory(self, xinput=None, train=False, inf_iter=None, ascent=True):
    tflearn.is_training(is_training=train, session=self.sess)
    h_init = np.random.normal(0, 1, size=(np.shape(xinput)[0], self.config.hidden_num))
    feeddic = {self.x: xinput,
                 self.h: h_init,
                 self.inf_penalty_weight_ph: self.config.inf_penalty,
                 self.is_training : 1.0 if train else 0.0,
                 self.bs_ph: np.shape(xinput)[0],
                 self.dropout_ph: self.config.dropout}
    h_ar = self.sess.run(self.h_ar ,feed_dict=feeddic)
    return h_ar

  def soft_predict(self, xinput=None, hstart=None, train=False, inf_iter=None, ascent=True, end2end=False):
    tflearn.is_training(is_training=train, session=self.sess)
    if end2end:
      #h_init = np.random.normal(0, 1, size=(np.shape(xinput)[0], self.config.hidden_num))
      #h_init = np.random.normal(0, 2, size=(np.shape(xinput)[0], self.config.hidden_num))
      h_init = np.zeros((np.shape(xinput)[0], self.config.hidden_num))
      feeddic = {self.x: xinput,
                 self.h: h_init,
                 self.is_training: 1.0 if train else 0.0,
                 self.inf_penalty_weight_ph: self.config.inf_penalty,
                 self.bs_ph: np.shape(xinput)[0],
                 self.dropout_ph: self.config.dropout}
      yp = self.sess.run(self.yp, feed_dict=feeddic)
    else:

      self.inf_objective = self.energy_h
      self.inf_gradient = self.energy_hgradient
      h_a = self.inference(xinput=xinput, hstart=hstart, inf_iter=inf_iter, train=train, ascent=ascent)
      return self.map_predict_h(xinput=xinput,hidden=np.average(h_a,0))


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
        yp_ar =  [np.argmax(yp, 2) for yp in soft_yp_ar]
      else:
        if self.config.verbose > 3:
          for k in range(self.config.inf_iter):
            print (np.average(en_ar[k]))
        yp_ar = soft_yp_ar
      return yp_ar
    else:
      raise NotImplementedError

  def map_predict_h(self, xinput=None, hidden=None):
    tflearn.is_training(False, self.sess)
    feeddic = {self.x: xinput,
               self.h: hidden,
               self.dropout_ph: self.config.dropout}
    yp = self.sess.run(self.yp_hpredict, feed_dict=feeddic)
    if self.config.dimension > 1:
      return np.argmax(yp, 2)
    else:
      return yp

  def map_predict(self, xinput=None, train=False, inf_iter=None, ascent=True, end2end=False, continuous=False):
    yp = self.soft_predict(xinput=xinput, train=train, inf_iter=inf_iter, ascent=ascent, end2end=end2end)
    if self.config.dimension == 1:
      return yp
    else:
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
    if self.config.dimension > 1:
      yt_ind = self.var_to_indicator(ybatch)
      yt_ind = np.reshape(yt_ind, (-1, self.config.output_num*self.config.dimension))
    #xd, yd, yp_ind = self.get_all_diff(xinput=xbatch, yinput=yt_ind, ascent=True, inf_iter=10)
    else:
      yt_ind = ybatch

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

  # def train_supervised_e2e_batch3(self, xbatch, ybatch, verbose=0):
  #   tflearn.is_training(True, self.sess)
  #   if self.config.dimension > 1:
  #     yt_ind = self.var_to_indicator(ybatch)
  #     yt_ind = np.reshape(yt_ind, (-1, self.config.output_num*self.config.dimension))
  #   else:
  #     yt_ind = ybatch
  #
  #   h_init = np.random.normal(0, 1, size=(np.shape(xbatch)[0], self.config.hidden_num))
  #   #h_0 = np.zeros((np.shape(xbatch)[0], self.config.hidden_num))
  #   feeddic = {self.x:xbatch, self.yt_ind: yt_ind,
  #              self.h: h_init,
  #              #self.h0 : h_0,
  #              self.learning_rate_ph:self.config.learning_rate,
  #              self.inf_penalty_weight_ph: self.config.inf_penalty,
  #              self.is_training: 1.0,
  #              self.dropout_ph: self.config.dropout}
  #
  #   if self.train_iter < self.config.pretrain_iter:
  #     _, o,en_ar, g_ar, h_ar  = self.sess.run([self.train_pre_step, self.obj_mlp, self.en_ar, self.g_ar, self.h_ar], feed_dict=feeddic)
  #
  #   else:
  #     _, o,en_ar, g_ar, h_ar  = self.sess.run([self.train_post_step, self.objective, self.en_ar, self.g_ar, self.h_ar], feed_dict=feeddic)
  #
  #     if verbose > 0:
  #       print ("---------------------------------------------------------")
  #       for k in range(self.config.inf_iter):
  #         print (np.average(np.linalg.norm(g_ar[k], axis=1)), np.average(np.linalg.norm(h_ar[k], axis=1)), np.average(en_ar[k]),)
  #   return o


  def train_supervised_e2e_batch_o(self, xbatch, ybatch, verbose=0):
    tflearn.is_training(True, self.sess)
    if self.config.dimension > 1:
      yt_ind = self.var_to_indicator(ybatch)
     # print(yt_ind[0,0:1000,0])
     # print(yt_ind[0,0:1000,1])
      yt_ind = np.reshape(yt_ind, (-1, self.config.output_num*self.config.dimension))

    else:
      yt_ind = ybatch

    h_init = np.random.normal(0, self.config.scale, size=(np.shape(xbatch)[0], self.config.hidden_num))
    #h_0 = np.zeros((np.shape(xbatch)[0], self.config.hidden_num))
    feeddic = {self.x:xbatch, self.yt_ind: yt_ind,
               self.h: h_init,
               #self.h0 : h_0,
               self.learning_rate_ph:self.config.learning_rate,
               self.inf_penalty_weight_ph: self.config.inf_penalty,
               self.is_training: 1.0,
               self.dropout_ph: self.config.dropout}


    _, o, g, h, iter = self.sess.run(
          [self.train_all_step, self.objective, self.g_ar, self.h_ar, self.iter], feed_dict=feeddic)
    if verbose > 0:
      print (o, g[-1], h[-1], iter)
    return o

  def train_supervised_e2e_batch(self, xbatch, ybatch, verbose=0):
    tflearn.is_training(True, self.sess)
    if self.config.dimension > 1:
      yt_ind = self.var_to_indicator(ybatch)
     # print(yt_ind[0,0:1000,0])
     # print(yt_ind[0,0:1000,1])
      yt_ind = np.reshape(yt_ind, (-1, self.config.output_num*self.config.dimension))

    else:
      yt_ind = ybatch

    h_init = np.random.normal(0, self.config.scale, size=(np.shape(xbatch)[0], self.config.hidden_num))
    #h_0 = np.zeros((np.shape(xbatch)[0], self.config.hidden_num))
    feeddic = {self.x:xbatch, self.yt_ind: yt_ind,
               self.h: h_init,
               self.bs_ph : np.shape(xbatch)[0],
               #self.h0 : h_0,
               self.learning_rate_ph:self.config.learning_rate,
               self.inf_penalty_weight_ph: self.config.inf_penalty,
               self.is_training: 1.0,
               self.dropout_ph: self.config.dropout}

    if False and self.train_iter % 2 == 0 : # < self.config.pretrain_iter:
      _, o = self.sess.run([self.train_pred_step, self.objective], feed_dict=feeddic)

    else:
      if self.config.pretrain_iter < 0:
        _, o, en_ar, g_ar, h_ar, l_ar = self.sess.run(
          [self.train_all_step, self.objective, self.en_ar, self.g_ar, self.h_ar, self.l_ar], feed_dict=feeddic)

      else:
        _, o,en_ar, g_ar, h_ar, l_ar  = self.sess.run([self.train_step, self.objective, self.en_ar,
                                                       self.g_ar, self.h_ar, self.l_ar], feed_dict=feeddic)

      if verbose > 0:
        print ("---------------------------------------------------------")
        for k in range(self.config.inf_iter):
          if verbose>1:
            print (h_ar[k][0])
          print (g_ar[k], np.average(np.sum(h_ar[k], axis=1)), np.average(en_ar[k]), np.average(l_ar[k]) )
    return o


  def train_supervised_e2e_batch4(self, xbatch, ybatch, verbose=0):
    tflearn.is_training(True, self.sess)
    if self.config.dimension > 1:
      yt_ind = self.var_to_indicator(ybatch)
     # print(yt_ind[0,0:1000,0])
     # print(yt_ind[0,0:1000,1])
      yt_ind = np.reshape(yt_ind, (-1, self.config.output_num*self.config.dimension))

    else:
      yt_ind = ybatch

    h_init = np.random.normal(0, self.config.scale, size=(np.shape(xbatch)[0], self.config.hidden_num))
    #h_0 = np.zeros((np.shape(xbatch)[0], self.config.hidden_num))
    feeddic = {self.x:xbatch, self.yt_ind: yt_ind,
               self.h: h_init,
               self.learning_rate_ph:self.config.learning_rate,
               self.inf_penalty_weight_ph: self.config.inf_penalty,
               self.is_training: 1.0,
               self.dropout_ph: self.config.dropout}

    if self.train_iter < self.config.pretrain_iter:
      _, o = self.sess.run([self.train_pre_step, self.obj_mlp], feed_dict=feeddic)

    else:
      _, o,en_ar, g_ar, h_ar, l_ar  = self.sess.run([self.train_post_step, self.objective, self.en_ar, self.g_ar, self.h_ar, self.l_ar], feed_dict=feeddic)

      if verbose > 0:
        print ("---------------------------------------------------------")
        for k in range(self.config.inf_iter):
          print (h_ar[0])

          print (np.average(np.linalg.norm(g_ar[k], axis=1)), np.mean(h_ar[k]), np.average(en_ar[k]), np.average(l_ar[k]) )
    return o


  # def train_supervised_e2e_batch2(self, xbatch, ybatch, verbose=0):
  #   tflearn.is_training(True, self.sess)
  #   if self.config.dimension > 1:
  #     yt_ind = self.var_to_indicator(ybatch)
  #     yt_ind = np.reshape(yt_ind, (-1, self.config.output_num*self.config.dimension))
  #
  #   else:
  #     yt_ind = ybatch
  #
  #   h_init = np.random.normal(0, 1, size=(np.shape(xbatch)[0], self.config.hidden_num))
  #   #h_0 = np.zeros((np.shape(xbatch)[0], self.config.hidden_num))
  #   feeddic = {self.x:xbatch, self.yt_ind: yt_ind,
  #              self.h: h_init,
  #              #self.h0 : h_0,
  #              self.learning_rate_ph:self.config.learning_rate,
  #              self.inf_penalty_weight_ph: self.config.inf_penalty,
  #              self.is_training: 1.0,
  #              self.dropout_ph: self.config.dropout}
  #
  #
  #
  #
  #   if self.train_iter < self.config.pretrain_iter:
  #     _, o = self.sess.run([self.train_hspen_step, self.objective], feed_dict=feeddic)
  #
  #   else:
  #     _, o = self.sess.run(
  #         [self.train_predhx_step, self.objective], feed_dict=feeddic)
  #   return o

  def save(self, path):
    self.saver.save(self.sess, path)

  def restore(self, path):
    self.saver.restore(self.sess, path)
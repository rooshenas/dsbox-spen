
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

class HSPEN:
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


  def construct(self, training_type = TrainingType.Rank_Based ):
    if training_type == TrainingType.SSVM:
      return self.ssvm_training()
    elif training_type == TrainingType.Rank_Based:
      return self.rank_training()
    else:
      raise NotImplementedError



  def ll_training(self):
    #self.h = tf.placeholder(tf.float32, shape=[None, self.config.hidden_num])
    #self.h2 = tf.placeholder(tf.float32, shape=[None, self.config.hidden_num])
    self.yp = tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension])
    self.yt = tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension])



    obj1 = tf.reduce_sum(tf.maximum((self.ce2 - self.ce1) * self.margin_weight_ph - self.spen_y1 + self.spen_y2, 0.0))



  def softmax_prediction_network(self, hinput, reuse=False):
    net = hinput
    with tf.variable_scope(self.config.spen_variable_scope):
      with tf.variable_scope("pred") as scope:
        #net = tflearn.fully_connected(net, 1000, activation='relu',
        #                              weight_decay=self.config.weight_decay,
        #                              weights_init=tfi.variance_scaling(),
        #                              bias_init=tfi.zeros(), reuse=reuse,
        #                              scope=("ph.0"))
        net = tflearn.fully_connected(net, self.config.output_num*self.config.dimension, activation='linear',
                                      weight_decay=self.config.weight_decay,
                                      weights_init=tfi.variance_scaling(),
                                      bias_init=tfi.zeros(), reuse=reuse,
                                      scope=("ph.1"))

    #cat_output = tf.reshape(net, (-1, self.config.output_num, self.config.dimension))
    #return tf.nn.so(cat_output, dim=2)
    return net


  def rank_training(self):
    self.h1 = tf.placeholder(tf.float32, shape=[None, self.config.hidden_num], name="Hidden1")
    self.h2 = tf.placeholder(tf.float32, shape=[None, self.config.hidden_num], name="Hidden2")
    self.y1 = tf.placeholder(tf.float32, shape=[None, self.config.output_num,self.config.dimension], name="YOutput")
    self.y2 = tf.placeholder(tf.float32, shape=[None, self.config.output_num,self.config.dimension], name="YOutput2")
    self.margin_weight_ph = tf.placeholder(tf.float32, shape=[], name="Margin")


    self.yp1 = self.prediction_net(hidden_vars=self.h1)
    self.yp2 = self.prediction_net(hidden_vars=self.h2, reuse=True)
    #self.yp2 = self.yp1
    flat_yp1 = tf.reshape(self.yp1, shape=(-1, self.config.output_num*self.config.dimension))
    flat_yp2 = tf.reshape(self.yp2, shape=(-1, self.config.output_num*self.config.dimension))

    self.spen_y1 = self.get_energy(xinput=self.x, yinput=self.h1, embedding=self.embedding)
    self.spen_y2 = self.get_energy(xinput=self.x, yinput=self.h2, embedding=self.embedding, reuse=True)


    ent_yp1 = -tf.reduce_sum(flat_yp1 * tf.log(tf.maximum(flat_yp1, 1e-20)), 1)
    ent_yp2 = -tf.reduce_sum(flat_yp2 * tf.log(tf.maximum(flat_yp2, 1e-20)), 1)

    self.spen = self.spen_y1
    self.h = self.h1
    self.yp = self.yp1
    self.y = self.y1

    self.spen_gradient = tf.gradients(self.spen, self.h)[0]

    flat_y1 = tf.reshape(self.y1, shape=(-1, self.config.output_num*self.config.dimension))
    flat_y2 = tf.reshape(self.y2, shape=(-1, self.config.output_num*self.config.dimension))
    self.ce1 = -tf.reduce_sum(flat_y1 * tf.log(tf.maximum(flat_yp1, 1e-20)), 1)
    self.ce2 = -tf.reduce_sum(flat_y1 * tf.log(tf.maximum(flat_yp2, 1e-20)), 1)

    #vloss = self.get_l2_loss()

    obj1 = tf.reduce_sum(tf.maximum((self.ce2 - self.ce1) * self.margin_weight_ph - self.spen_y1 + self.spen_y2, 0.0))
    self.v1_sum = tf.reduce_sum(self.ce1)
    self.v2_sum = tf.reduce_sum(self.ce2)
    self.e1_sum = tf.reduce_sum(self.spen_y1)
    self.e2_sum = tf.reduce_sum(self.spen_y2)
    self.objective = obj1 + 1.0 * tf.reduce_sum(ent_yp1+ ent_yp2)#+ self.config.l2_penalty * vloss  # + obj2
    self.num_update = tf.reduce_sum(
      tf.cast((self.ce2 - self.ce1) * self.margin_weight_ph >= (self.spen_y1 - self.spen_y2), tf.float32))
    ce = tf.reduce_sum(self.y * tf.log(tf.maximum(self.yp, 1e-20)), 1)
    self.pred_obj = -tf.reduce_sum(ce)

    #self.pred_train_step = tf.train.AdamOptimizer(self.learning_rate_ph).minimize(self.pred_obj,  var_list=[self.pred_variables(), self.fnet_variables()])
    self.train_step = tf.train.AdamOptimizer(self.learning_rate_ph).minimize(self.objective)

    return self

  def eval(self, xd, yp, yt):
    raise NotImplementedError

  def inference(self, xd, yt=None, inf_iter = None, train=True, ascent=True,
                initialization = InfInit.Random_Initialization):
    """
      ARGS:
        xd: Input Tensor
        yt: Ground Truth Output

      RETURNS:
        An array of Tensor of shape (-1, output_num, dimension)

    """
    if inf_iter is None:
      inf_iter = self.config.inf_iter
    tflearn.is_training(is_training=train, session=self.sess)
    bs = np.shape(xd)[0]

    if initialization == InfInit.Random_Initialization:
      hd = np.random.uniform(0,1.0, (bs, self.config.hidden_num))
    else:
      raise NotImplementedError("Other initialization methods are not supported.")

    i=0
    h_a = []
    while i < inf_iter:
      g = self.sess.run(self.inf_gradient, feed_dict={self.x:xd, self.h:hd, self.dropout_ph: self.config.dropout})
      #print (g), self.config.inf_rate, self.config

      if ascent:
        hd = hd + self.config.inf_rate * g
      else:
        hd = hd - self.config.inf_rate * g
      h_a.append(hd)
      i += 1

    return np.array(h_a)


  def get_first_large_consecutive_diff(self, xd, ascent=True, yt=None):
    self.inf_objective = self.spen
    self.inf_gradient = self.spen_gradient


    h_a  = self.inference( xd, train=True, ascent=ascent, inf_iter=self.config.inf_iter)
    h_a = np.flip(h_a, axis=0)

    en_a = np.array([self.sess.run(self.spen, feed_dict={self.x: xd,
                self.h: np.reshape(h_i, (-1,self.config.hidden_num)),
                self.dropout_ph: self.config.dropout}) for h_i in h_a ])
    yp_a = np.array([self.h_map_predict(h_i) for h_i in h_a])



    f_a = np.array([self.eval(xd, y_i, yt=yt) for y_i in yp_a])
    size = np.shape(xd)[0]
    t = np.array(range(size))
    f1 = []
    f2 = []
    h1 = []
    h2 = []
    x = []
    k = 0
    y = []
    traces=[]

    while k < (len(f_a) - 1) and len(t) > 0:
      fa_s = f_a[k:k+2,:]
      en_s = en_a[k:k+2,:]
      h_s = h_a[k:k+2, :, :]
      i_1 = np.argmax(fa_s, 0)
      i_2 = np.argmin(fa_s, 0)
      violatations = np.array([fa_s[i_1[i],i] - fa_s[i_2[i],i] - en_s[i_1[i],i] + en_s[i_2[i],i] for i in t])
      ti = np.array(range(len(t)))
      tk = ti[np.where(violatations[ti]>0)]
      traces.append(len(tk))
      for i in tk:
        f1.append(fa_s[i_1[t[i]],t[i]])
        f2.append(fa_s[i_2[t[i]],t[i]])
        h1.append(h_s[i_1[t[i]],t[i],:])
        h2.append(h_s[i_2[t[i]],t[i],:])
        x.append(xd[t[i],:])
        if yt is not None:
          y.append(yt[t[i],:])
      tv = ti[np.where(violatations[ti]<=0)]
      t = t[tv]
      k = k + 1

    #y1 = self.flatten_categorial(np.argmax(y1, 2))
    #y2 = self.flatten_categorial(np.argmax(y2, 2))

    x = np.array(x)
    f1 = np.array(f1)
    f2 = np.array(f2)
    y = np.array(y)
    return x, y, h1, h2, f1, f2


  def set_train_iter(self, iter):
    self._train_iter = iter

  def get_train_iter(self):
    return self._train_iter

  def prediction_net(self, hidden_vars=None, reuse=False ):
    raise NotImplementedError

  def get_energy(self, xinput=None, yinput=None, embedding=None, reuse=False):
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
    return np.reshape(cat, (size[0], self.config.output_num , self.config.dimension))

  def indicator_to_var(self, id):
    size = np.shape(id)
    y_cat_indicator = np.reshape(id, (size[0], self.config.output_num, self.config.dimension))
    y_m = np.argmax(y_cat_indicator, 2)
    return y_m




  def soft_predict(self, hd, train=False, inf_iter=None):
    #hd = self.hpredict(xd, train=train, inf_iter=inf_iter)
    #print(hd)
    yp = self.sess.run(self.yp, feed_dict={self.h: hd, self.dropout_ph: self.config.dropout})
    return np.reshape(yp, newshape=(-1, self.config.output_num, self.config.dimension))


  def map_predict(self, xd, train=False, inf_iter=None, ascent=True):
    hp = self.hpredict(xd, inf_iter=inf_iter, train=train, ascent=ascent)
    yp = self.soft_predict(hp)
    return np.argmax(yp, 2)

  def h_map_predict(self, hd):
    yp = self.soft_predict(hd)
    return np.argmax(yp, 2)

  def hpredict(self, xd=None, inf_iter=None, train=False, ascent=True ):
    self.inf_objective = self.spen
    self.inf_gradient = self.spen_gradient
    h_a = self.inference(xd, inf_iter=inf_iter, train=train, ascent=ascent)
    return h_a[-1]


  def predict_cl_trajectory(self, xd=None, yt=None, inf_iter=None, train=False, ascent=True):
    self.inf_objective = self.spen
    self.inf_gradient = self.spen_gradient
    h_a = self.inference(xd, inf_iter=inf_iter, train=train, ascent=ascent)
    yp_a = np.array([np.minimum(self.soft_predict(hp), yt) for hp in h_a])
    return yp_a


  def train_batch(self, xbatch=None, ybatch=None):
    raise NotImplementedError

  def train_rank_supervised_batch(self, xbatch, ybatch, verbose=0):
    ybatch_ind = self.var_to_indicator(ybatch)
    if self._train_iter > 1:
      for j in range(-1):
        _, o2 = self.sess.run([self.pred_train_step, self.pred_obj],
                              feed_dict={self.x: xbatch, self.y: ybatch_ind, self.h: self.hpredict(xbatch),
                                         self.learning_rate_ph: self.config.learning_rate,
                                         self.dropout_ph: self.config.dropout })
        print (j,o2)



    xd, yd, h1, h2, f1, f2 = self.get_first_large_consecutive_diff(xbatch, yt=ybatch, ascent=True)


    if np.shape(xd)[0] < 2 :
      return -1,-1
    #yp = self.soft_predict(hd=hp, train=True, inf_iter=10)
    #print("here 2")
    #yp_flat = np.reshape(yp, newshape=(-1, self.config.output_num* self.config.dimension))
    yd_ind = self.var_to_indicator(yd)
    feeddic={
      self.x: xd,
      self.h1: h1,
      self.h2: h2,
      self.y: yd_ind,
      self.learning_rate_ph: self.config.learning_rate,
      self.dropout_ph: self.config.dropout,
      self.margin_weight_ph: self.config.margin_weight
    }
    _, obj, n, v1, v2, e1, e2, yp = self.sess.run([self.train_step, self.objective,
                                               self.num_update,self.v1_sum, self.v2_sum,
                                               self.e1_sum, self.e2_sum, self.yp1],
                                              feed_dict=feeddic )
    print (obj,n, v1,v2,e1,e2)
    #print ("yp:", yp[0,0,:])
    #print ("yt:", yd_ind[0,0,:])
    return obj,  np.shape(xd)[0]



import tensorflow as tf
import numpy as np
import math
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

  def mse_loss(self, yt, yp):
    l = tf.reduce_mean(tf.reduce_sum(tf.square(yt - yp),1))
    return l

  def ce_loss(self, yt, yp):
    eps=1e-30
    l = -tf.reduce_sum(yt * tf.log(tf.maximum(yp, eps)),1) - tf.reduce_sum(1-yt * tf.log(tf.maximum(1-yp, eps)),1)
    return l

  def sym_f1(self, yt, yp):
    yp = tf.reshape(yp, [-1, self.config.output_num, self.config.dimension])
    yt = tf.reshape(yt, [-1, self.config.output_num, self.config.dimension])
    #yp_zeros = 1.0 - tf.squeeze(yp[:, :, 0])
    yp_ones = yp[:,:,1]
    yt_ones = yt[:,:,1]
    intersect = tf.reduce_sum(tf.minimum  (yt_ones, yp_ones),1)
    return -tf.reduce_sum(intersect) + self.ce_loss(yt, yp)
    #return (self.f1_loss(yt_ones, yp_ones)) #+ self.f1_loss(yt_zeros, yp_zeros))/2.0

  def f1_loss(self, yt, yp):
    intersect = tf.reduce_sum(tf.minimum(yt, yp),1)
    union = tf.reduce_sum(tf.maximum(yt, yp),1)
    return 1-tf.reduce_sum(2*intersect / (union+intersect))

  def get_loss(self, yt, yp):
    raise NotImplementedError

  def biased_loss(self, yt, yp):
    l = -tf.reduce_sum ((tf.reduce_sum(yt * tf.log(tf.maximum(yp, 1e-20)), 1) \
       + tf.reduce_sum((1. - yt) * tf.log(tf.maximum(1. - yp , 1e-20)), 1)) )
    yp = tf.reshape(yp, [-1, self.config.output_num, self.config.dimension])
    yp_zeros = yp[:, :, 0]
    yp_ones = yp[:,:, 1]
    en = tf.reduce_sum(yp_ones * tf.log(yp_ones), 1)
    return l + 1.2*(tf.reduce_sum(yp_zeros) - tf.reduce_sum(yp_ones)) +  0.0*tf.reduce_sum(en)

  def iou_loss(self, yt, yp):
    #code from here
    #http://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html
    #logits = tf.reshape(yp, [-1])
    #trn_labels = tf.reshape(yt, [-1])
    logits = yt
    trn_labels = yp
    '''
    Eq. (1) The intersection part - tf.mul is element-wise, 
    if logits were also binary then tf.reduce_sum would be like a bitcount here.
    '''
    inter = tf.reduce_sum(tf.multiply(logits, trn_labels),1)

    '''
    Eq. (2) The union part - element-wise sum and multiplication, then vector sum
    '''
    union = tf.reduce_sum(tf.subtract(tf.add(logits, trn_labels), tf.multiply(logits, trn_labels)),1)

    # Eq. (4)
    loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.div(inter, union))

    return loss #tf.div(inter, union)


  def get_initialization_net(self, xinput, output_size, embedding=None, reuse=False):
    raise NotImplementedError

  def get_loss_network(self, ytrue=None, ypred=None, reuse=False):
    raise NotImplementedError

  def get_true_loss(self, ytrue=None, ypred=None):
    raise NotImplementedError



  def value_training(self):
    self.inf_penalty_weight_ph = tf.placeholder(tf.float32, shape=[], name="InfPenalty")
    self.yt_ind= tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYT")
    self.yp_ind= tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYP")
    y_start, features = self.get_initialization_net(self.x, self.config.output_num * self.config.dimension, embedding=self.embedding)



    current_yp_ind = y_start
    self.objective = 0.0
    self.yp_ar = []
    self.l_ar = []
    self.g_ar = []
    self.en_ar = []
    self.ind_ar = []
    self.v_ar = []
    self.ld_ar = []

    if self.config.dimension == 1:
      yp_ind = tf.nn.sigmoid(current_yp_ind)#tf.clip_by_value(current_yp_ind, clip_value_min=0.0, clip_value_max=1.0)
    else:
      yp_ind = tf.nn.softmax(current_yp_ind)

    for i in range(int(self.config.inf_iter)):
      penalty_current = 0.1*tf.reduce_sum(tf.square(current_yp_ind-y_start),1)
      self.energy_y = self.get_energy(xinput=features, yinput=yp_ind, embedding=self.embedding, reuse=True if i > 0 else False) - penalty_current
      g = tf.gradients(self.energy_y, current_yp_ind)[0]
      g = tf.clip_by_value(g, clip_value_min=-1.0, clip_value_max=1.0)
      next_yp_ind = current_yp_ind + self.config.inf_rate * g
      current_yp_ind = next_yp_ind
      if self.config.dimension > 1:
        yp_matrix = tf.reshape(current_yp_ind, [-1, self.config.output_num, self.config.dimension])
        l = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
          logits=tf.reshape(current_yp_ind,(-1, self.config.output_num, self.config.dimension)),
          labels=tf.reshape(self.yt_ind, (-1, self.config.output_num, self.config.dimension))))

        yp_current = tf.nn.softmax(yp_matrix, 2)
      else:
        #yp_current = tf.clip_by_value(current_yp_ind, clip_value_min=0.0, clip_value_max=1.0)#
        l_direct = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.yt_ind, logits=current_yp_ind))
        yp_current =  tf.nn.sigmoid(current_yp_ind)




      yp_ind = tf.reshape(yp_current, [-1, self.config.output_num * self.config.dimension])
      #l_direct = self.get_loss(self.yt_ind, yp_ind)

      l = self.get_loss_network(ytrue=self.yt_ind, ypred=yp_ind, reuse=True if i > 0 else False)

      self.ind_ar.append(tf.reduce_mean(tf.norm(yp_current,1)))
      self.l_ar.append(tf.reduce_mean(l))
      self.en_ar.append(self.energy_y)
      self.g_ar.append(tf.reduce_mean(tf.norm(g,1)))
      #self.objective = (1-self.config.alpha)*self.objective + self.config.alpha * l
      #self.objective += (self.config.alpha / (self.config.inf_iter - i+1.0)) * l
      self.yp_ar.append(yp_current)
      v = self.iou_loss(self.yt_ind, yp_ind)
      self.v_ar.append(v)
      self.ld_ar.append(l_direct)

    #self.objective += self.config.l2_penalty * self.get_l2_loss()

    self.objective = -tf.reduce_sum ((tf.reduce_sum(v * tf.log(tf.maximum(l, 1e-20)), 1) \
       + tf.reduce_sum((1. - v) * tf.log(tf.maximum(1. - l , 1e-20)), 1)) ) + tf.reduce_sum(l)
    self.yp = self.yp_ar[-1] #self.get_prediction_net(input=self.h_state)
    #l = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.yt_ind, logits=current_yp_ind))
    self.objective += self.config.l2_penalty * self.get_l2_loss()


    #self.yp_ind = tf.reshape(self.yp, [-1, self.config.output_num * self.config.dimension], name="reshaped")
    #self.objective = -tf.reduce_sum(self.yt_ind * tf.log( tf.maximum(self.yp_ind, 1e-20)))
    self.train_step = self.optimizer.minimize(self.objective)



  def direct_loss_training(self):
    self.inf_penalty_weight_ph = tf.placeholder(tf.float32, shape=[], name="InfPenalty")
    self.yt_ind= tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYT")
    self.yp_ind= tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYP")


    y_start, features = self.get_initialization_net(self.x, self.config.output_num * self.config.dimension, embedding=self.embedding)

    current_yp_ind = y_start
    self.objective = 0.0
    self.yp_ar = []
    self.l_ar = []
    self.g_ar = []
    self.en_ar = []
    self.ind_ar = []

    if self.config.dimension == 1:
      yp_ind = tf.nn.sigmoid(current_yp_ind)
    else:
      yp_ind = tf.nn.softmax(current_yp_ind)

    for i in range(int(self.config.inf_iter)):
      self.energy_y = self.get_energy(xinput=self.x, yinput=yp_ind, embedding=self.embedding, reuse=True if i > 0 else False)# - penalty_current
      g = tf.gradients(self.energy_y, current_yp_ind)[0]
      #g = tf.clip_by_value(g, clip_value_min=-1.0, clip_value_max=1.0)
      next_yp_ind = current_yp_ind + self.config.inf_rate * g
      current_yp_ind = next_yp_ind
      if self.config.dimension > 1:
        yp_matrix = tf.reshape(current_yp_ind, [-1, self.config.output_num, self.config.dimension])
        yp_current = tf.nn.softmax(yp_matrix, 2)
      else:
        yp_current =  tf.nn.sigmoid(current_yp_ind)

      yp_ind = tf.reshape(yp_current, [-1, self.config.output_num * self.config.dimension])
      l = tf.reduce_sum(self.get_loss(self.yt_ind, yp_ind))
      self.ind_ar.append(tf.reduce_mean(tf.norm(yp_current,1)))
      self.l_ar.append(l)
      self.en_ar.append(self.energy_y)
      self.g_ar.append(tf.reduce_mean(tf.norm(g,1)))
      self.yp_ar.append(yp_current)

    self.yp = self.yp_ar[-1] #self.get_prediction_net(input=self.h_state)
    self.objective =  l + self.config.l2_penalty * self.get_l2_loss()


    self.train_step = self.optimizer.minimize(self.objective)




  def end2end_training(self):
    self.inf_penalty_weight_ph = tf.placeholder(tf.float32, shape=[], name="InfPenalty")
    self.yt_ind= tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYT")
    self.yp_ind= tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYP")
    #y_start, features = self.get_initialization_net(self.x, self.config.output_num * self.config.dimension, embedding=self.embedding)

    y_start = self.yp_ind

    current_yp_ind = y_start
    self.objective = 0.0
    self.yp_ar = []
    self.l_ar = []
    self.g_ar = []
    self.en_ar = []
    self.ind_ar = []

    if self.config.dimension == 1:
      yp_ind = tf.nn.sigmoid(current_yp_ind)#tf.clip_by_value(current_yp_ind, clip_value_min=0.0, clip_value_max=1.0)
    else:
      yp_ind = tf.nn.softmax(current_yp_ind)

    for i in range(int(self.config.inf_iter)):
      #penalty_current = 0.1*tf.reduce_sum(tf.square(current_yp_ind-y_start),1)
      self.energy_y = self.get_energy(xinput=self.x, yinput=yp_ind, embedding=self.embedding, reuse=True if i > 0 else False)# - penalty_current
      g = tf.gradients(self.energy_y, current_yp_ind)[0]
      #g = tf.clip_by_value(g, clip_value_min=-1.0, clip_value_max=1.0)
      next_yp_ind = current_yp_ind + self.config.inf_rate * g
      current_yp_ind = next_yp_ind
      if self.config.dimension > 1:


        yp_matrix = tf.reshape(current_yp_ind, [-1, self.config.output_num, self.config.dimension])
        l = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
          logits=tf.reshape(current_yp_ind,(-1, self.config.output_num, self.config.dimension)),
          labels=tf.reshape(self.yt_ind, (-1, self.config.output_num, self.config.dimension))))

        yp_current = tf.nn.softmax(yp_matrix, 2)
      else:
        #yp_current = tf.clip_by_value(current_yp_ind, clip_value_min=0.0, clip_value_max=1.0)#
        #l = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.yt_ind, logits=current_yp_ind))

        yp_current =  tf.nn.sigmoid(current_yp_ind)

      yp_ind = tf.reshape(yp_current, [-1, self.config.output_num * self.config.dimension])
      l = tf.reduce_sum(self.get_loss(self.yt_ind, yp_ind))
        #self.get_loss(self.yt_ind, yp_ind)
      self.ind_ar.append(tf.reduce_mean(tf.norm(yp_current,1)))
      self.l_ar.append(l)
      self.en_ar.append(self.energy_y)
      self.g_ar.append(tf.reduce_mean(tf.norm(g,1)))
      #self.objective = (1-self.config.alpha)*self.objective + self.config.alpha * l
      #self.objective += (self.config.alpha / (self.config.inf_iter - i+1.0)) * l
      self.yp_ar.append(yp_current)

    #self.objective += self.config.l2_penalty * self.get_l2_loss()


    self.yp = self.yp_ar[-1] #self.get_prediction_net(input=self.h_state)
    #l = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.yt_ind, logits=current_yp_ind))
    self.objective =  l + self.config.l2_penalty * self.get_l2_loss()


    #self.yp_ind = tf.reshape(self.yp, [-1, self.config.output_num * self.config.dimension], name="reshaped")
    #self.objective = -tf.reduce_sum(self.yt_ind * tf.log( tf.maximum(self.yp_ind, 1e-20)))
    self.train_step = self.optimizer.minimize(self.objective)



  def ssvm_training(self):
    self.margin_weight_ph = tf.placeholder(tf.float32, shape=[], name="Margin")
    self.yp = tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYP")
    self.yt = tf.placeholder(tf.float32, shape=[None, self.config.output_num * self.config.dimension], name="OutputYT")



    self.energy_yp = self.get_energy(xinput=self.x, yinput=self.yp, embedding=self.embedding)
    self.energy_yt = self.get_energy(xinput=self.x, yinput=self.yt, embedding=self.embedding, reuse=True)

    self.ce = -tf.reduce_sum(self.yt * tf.log( tf.maximum(self.yp, 1e-20)), 1) \
              + -tf.reduce_sum((1.0-self.yt) * tf.log( tf.maximum((1-self.yp), 1e-20)), 1)
    #self.loss_augmented_energy = self.energy_yp - self.ce
    self.loss_augmented_energy = self.energy_yp + self.ce #+ tf.log(1e-1+tf.reduce_sum(tf.square(self.yt - self.yp),1))
    self.loss_augmented_energy_ygradient = tf.gradients(self.loss_augmented_energy, self.yp)[0]

    self.energy_ygradient = tf.gradients(self.energy_yp, self.yp)[0]

    self.objective = tf.reduce_sum( tf.maximum( self.energy_yp + self.margin_weight_ph - self.energy_yt, 0.0)) \
                     + self.config.l2_penalty * self.get_l2_loss()

    self.num_update = tf.reduce_sum(tf.cast( self.ce * self.margin_weight_ph > self.energy_yt - self.energy_yp, tf.float32))
    #self.num_update = tf.reduce_sum(tf.cast(0.0 > self.energy_yt - self.loss_augmented_energy, tf.float32))
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
    elif training_type == TrainingType.Value_Matching:
      return self.value_training()
    else:
      raise NotImplementedError

  def gather_numpy(self, y, dim, index):
      """
      https://stackoverflow.com/questions/46065873/how-to-do-scatter-and-gather-operations-in-numpy
      Gathers values along an axis specified by dim.
      For a 3-D tensor the output is specified by:
          out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
          out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
          out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

      :param dim: The axis along which to index
      :param index: A tensor of indices of elements to gather
      :return: tensor of gathered values
      """
      idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
      self_xsection_shape = y.shape[:dim] + y.shape[dim + 1:]
      if idx_xsection_shape != self_xsection_shape:
          raise ValueError("Except for dimension " + str(dim) +
                           ", all dimensions of index and self should be the same size")
      if index.dtype != np.dtype('int_'):
          raise TypeError("The values of index must be integers")
      data_swaped = np.swapaxes(y, 0, dim)
      index_swaped = np.swapaxes(index, 0, dim)
      gathered = np.choose(index_swaped, data_swaped)
      return np.swapaxes(gathered, 0, dim)

  def project_simplex_opt(self, y):
      y = np.reshape(y, (-1, self.config.output_num, self.config.dimension))
      u = np.flip(np.sort(y, axis=-1), axis=-1)
      positions = np.arange(0, u.shape[-1])
      u_sum = (1 - u.cumsum(axis=-1)) / (positions + 1.0)
      ru = np.argmax((((u_sum + u > 0) + 0) * (positions + 1)), axis=-1)
      ru_ind = np.reshape(np.repeat(ru, self.config.dimension, axis=-1), y.shape)
      lambda_ = self.gather_numpy(u_sum, 2, ru_ind)
      yd_norm= np.clip(y + lambda_, a_min=0.0, a_max=1e1000)
      return np.reshape(yd_norm, (-1, self.config.output_num * self.config.dimension))

  def project_simplex_norm(self, y_ind):

    dim = self.config.dimension
    if dim > 1:
      yd = np.reshape(y_ind, (-1, self.config.output_num, dim))
      eps = np.full(shape=np.shape(yd), fill_value=1e-10)
      y_min = np.min(yd, axis=2)
      y_min_all = np.reshape(np.repeat(y_min, dim), (-1, self.config.output_num, dim))
      yd_pos = yd - y_min_all
      yd_sum = np.reshape(np.repeat(np.sum(yd_pos,2),dim), (-1, self.config.output_num ,dim))
      yd_sum = yd_sum + eps
      yd_norm = np.divide(yd_pos, yd_sum)
      return np.reshape(yd_norm, (-1, self.config.output_num*dim))
    else:
      return np.clip(y_ind, a_min=0.0 + 1e-10, a_max=1.0-1e-10)

  def project_simplex(self, y):
      def expand(tensor, dim=0):
          while tensor.dim() < y.dim():
              tensor = tensor.unsqueeze(dim)
          return tensor

      sorted, _ = y.sort(dim=-1, descending=True)
      positions = np.arange(0, sorted.shape[-1])
      summed = (1 - sorted.cumsum(dim=-1)) / (positions + 1.0)
      indicator = (sorted + ((summed > 0) + 0.0))
      idx, _ = (indicator * positions).max(dim=-1)
      lambda_ = summed[-1, expand(idx, dim=-1)]
      return (y + lambda_).clamp(min=0)


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


  def inference(self, yinit=None, xinput=None, yinput=None, inf_iter=None, ascent=True, train=False, loss_aug=False):
    if inf_iter is None:
      inf_iter = self.config.inf_iter
    #tflearn.is_training(is_training=train, session=self.sess)
    size = np.shape(xinput)[0]

    if yinput is not None:
      yt_ind = np.reshape(yinput, (-1, self.config.output_num*self.config.dimension))

    if loss_aug:
      yp_ind = yt_ind[:]
    else:
      if yinit is not None:
        yp_ind = yinit[:]
      else:
        yp_ind = np.random.uniform(0, 1, (size, self.config.output_num * self.config.dimension))
        #yp = self.softmax(np.reshape(yp_ind, (size, self.config.output_num, self.config.dimension)), axis=2)

        yp_ind = self.project_simplex_opt(yp_ind)
        #yp = np.zeros((size, self.config.output_num))
        #yp_ind = np.reshape(self.var_to_indicator(yp), (-1, self.config.output_num*self.config.dimension))


    if self.config.loglevel > 5:
      print ("inference:")
    #  print(yp_ind[0,:])
    i = 0
    yp_a = []
    self.g_ar = []
    self.en_ar = []
    #yp_ind2 = self.project_simplex_norm(yp_ind)
    #yp = np.reshape(yp_ind2, (-1, self.config.output_num, self.config.dimension))
    #yp_a.append(yp)
    mean = np.zeros(shape=np.shape(yp_ind))
    while i < inf_iter:

      yp_ind = self.project_simplex_opt(yp_ind)
      #yp = self.softmax(np.reshape(yp_ind, (size, self.config.output_num, self.config.dimension)), axis=2, theta=0.5)
      yp = np.reshape(yp_ind, (-1, self.config.output_num, self.config.dimension))
      yp_a.append(yp)
      if yinput is not None:
        feed_dict={self.x: xinput, self.yp: yp_ind, self.yt: yt_ind,
                    self.margin_weight_ph: self.config.margin_weight,
                    self.dropout_ph: self.config.dropout}
      else:
        feed_dict={self.x: xinput, self.yp: yp_ind,
                   self.margin_weight_ph: self.config.margin_weight,
                                     self.dropout_ph: self.config.dropout}

      g,en = self.sess.run([self.inf_gradient, self.inf_objective], feed_dict=feed_dict)
      gnorm = np.linalg.norm(g, axis=1)

      #g = np.clip(g,-10, 10)
      if train:
        noise = np.random.normal(mean, self.config.noise_rate*np.average(gnorm), size=np.shape(g))
      else:
        noise = np.zeros(shape=np.shape(g))
      if ascent:
        yp_ind = yp_ind + self.config.inf_rate * (g+noise)
      else:
        yp_ind = yp_ind - self.config.inf_rate * (g+noise)


      self.g_ar.append(g)
      self.en_ar.append(en)
      if self.config.loglevel > 10:
        print (g[0,0:50])


      if self.config.loglevel > 5:
          print("energy:", np.average(en), "yind:", np.average(np.sum(np.square(yp_ind), 1)),
                "gnorm:", np.average(gnorm), "yp:", np.average(np.max(yp, 2)))

      i += 1

    return np.array(yp_a)

  def evaluate(self, xinput=None, yinput=None, yt=None):
    raise NotImplementedError

  def search_better_y_fast(self, xtest, yprev):
    final_best = np.zeros((xtest.shape[0], self.config.output_num))
    for iter in range(np.shape(xtest)[0]):
      random_proposal = yprev[iter,:]
      score_first = self.evaluate(np.expand_dims(xtest[iter], 0), np.expand_dims(random_proposal, 0))
      start = score_first
      labelset = set(np.arange(self.config.dimension))
      found = False
      for l in range(self.config.output_num):
        for label in (labelset - set([yprev[iter,l]])): #set([random_proposal[l]]):
          random_proposal_new = random_proposal[:]
          random_proposal_new[l] = label
          score = self.evaluate(np.expand_dims(xtest[iter], 0),
                                np.expand_dims(random_proposal_new, 0))
          if score > score_first:
            score_first = score
            best_l = l
            best_label = label
            found = True
            #random_proposal[l] = random_proposal_new[l]
            #changed = True
            #break
      if self.config.loglevel > 4:
        print ("iter:", iter, "found:", found, "score first: ", start, "new score", score_first)
      final_best[iter, :] = yprev[iter, :]
      if found:
        final_best[iter, best_l] = best_label
    return final_best

  def get_first_large_consecutive_diff_new(self, xinput=None, yinput=None, inf_iter=None, ascent=True):
    self.inf_objective = self.energy_yp
    self.inf_gradient = self.energy_ygradient

    y_a = self.inference(xinput=xinput, train=True, ascent=ascent, inf_iter=inf_iter)


    en_a = np.array([self.sess.run(self.inf_objective,
                                   feed_dict={self.x: xinput,
                                              self.yp: np.reshape(y_i, (
                                                -1, self.config.output_num * self.config.dimension)),
                                              self.dropout_ph: self.config.dropout})
                     for y_i in y_a])
    ind = np.argmax(en_a, 0) if ascent else np.argmin(en_a, 0)
    yp = [y_a[ind[i], i, :] for i in range(np.shape(xinput)[0])]

    if np.random.random() > 0.8:
      print("search")
      y_better = self.search_better_y_fast(xinput, np.argmax(yp, 2))
      y_better = self.var_to_indicator(y_better)

      y_a = np.vstack((y_a, np.expand_dims(y_better, 0)))
      #y_a = y_a[-4:]

      en_p = np.array(self.sess.run(self.inf_objective,
                                   feed_dict={self.x: xinput,
                                              self.yp: np.reshape(yp, (
                                                -1, self.config.output_num * self.config.dimension)),
                                              self.dropout_ph: self.config.dropout}))
      en_a = np.vstack((en_a, en_p))

    f_a = np.array([self.evaluate(xinput=xinput, yinput=np.argmax(y_i, 2), yt=yinput) for y_i in y_a])

    # print (np.average(en_a, axis=1))
    # print (np.average(f_a, axis=1))
    if self.config.loglevel > 4:
      for t in range(xinput.shape[0]):
        print (t, f_a[-2][t], f_a[-1][t], np.argmax(yp,2)[t][:10])

    size = np.shape(xinput)[0]
    t = np.array(range(size))
    f1 = []
    f2 = []
    y1 = []
    y2 = []
    x = []
    k = 0
    it = np.shape(y_a)[0]
    for k in range(it - 1):
      for i in t:
        if f_a[k, i] > f_a[k + 1, i]:
          i_h = k
          i_l = k + 1
        else:
          i_l = k
          i_h = k + 1

        f_h = f_a[i_h, i]
        f_l = f_a[i_l, i]
        e_h = en_a[i_h, i]
        e_l = en_a[i_l, i]

        violation = (f_h - f_l) * self.config.margin_weight - e_h + e_l
        if violation > 0:
          f1.append(f_h)
          f2.append(f_l)
          y1.append((y_a[i_h, i, :]))
          y2.append((y_a[i_l, i, :]))
          x.append(xinput[i, :])

    x = np.array(x)
    f1 = np.array(f1)
    f2 = np.array(f2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    return x, y1, y2, f1, f2



  def get_first_large_consecutive_diff(self, xinput=None, yinput=None, inf_iter=None, ascent=True):
    self.inf_objective = self.energy_yp
    self.inf_gradient = self.energy_ygradient

    y_a = self.inference(xinput=xinput, train=True, ascent=ascent, inf_iter=inf_iter)
   # y_a = y_a[-5:]

    en_a = np.array([self.sess.run(self.inf_objective,
                feed_dict={self.x: xinput,
                           self.yp: np.reshape(y_i, (-1,self.config.output_num*self.config.dimension)),
                           self.dropout_ph: self.config.dropout})
                     for y_i in y_a ])
    f_a = np.array([self.evaluate(xinput=xinput, yinput=np.argmax(y_i,2), yt=yinput) for y_i in y_a])
    yp = y_a[-1]
    if self.config.loglevel > 4:
      for t in range(xinput.shape[0]):
        print(t, f_a[-2][t], f_a[-1][t], np.argmax(yp, 2)[t][:10])

    #print (np.average(en_a, axis=1))
    #print (np.average(f_a, axis=1))

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



  def soft_predict(self, yinit=None, xinput=None, train=False, inf_iter=None, ascent=True, end2end=False):
    #tflearn.is_training(is_training=train, session=self.sess)
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
      y_a = self.inference(yinit=yinit, xinput=xinput, inf_iter=inf_iter, train=train, ascent=ascent)
      en_a = np.array(self.en_ar)
      ind = np.argmax(en_a,0) if ascent else np.argmin(en_a, 0)
      yp = [y_a[ind[i],i,:] for i in range(np.shape(xinput)[0])]
      #yp = y_a[-1]

      if self.config.loglevel > 10:
        for i in range(int(inf_iter)):
          ym = np.argmax(y_a[i], -1)
          print (ym[0,0:20,])

      if self.config.dimension == 1:
        yp = np.reshape(yp, [-1, self.config.output_num])
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

      yp_ar = [np.argmax(yp, 2) if self.config.dimension > 1 else yp for yp in soft_yp_ar]
      return yp_ar
    else:
      raise NotImplementedError

  def map_predict(self, xinput=None, train=False, inf_iter=None, ascent=True, end2end=False):
    yp = self.soft_predict(xinput=xinput, train=train, inf_iter=inf_iter, ascent=ascent, end2end=end2end)
    if self.config.dimension == 1:

      return np.reshape(yp, (-1, self.config.output_num))
    else:
      return np.argmax(yp, 2)


  def loss_augmented_soft_predict(self, xinput=None, yinput=None, train=False, inf_iter=None, ascent=True, loss_aug=False):
    self.inf_objective = self.loss_augmented_energy
    self.inf_gradient = self.loss_augmented_energy_ygradient
    y_a = self.inference(xinput=xinput, yinput=yinput, inf_iter=inf_iter, train=train, ascent=ascent, loss_aug=loss_aug)
    if self.config.loglevel > 2:
      print (np.shape(self.g_ar), np.shape(y_a))
      for i in range(inf_iter):
        y = np.reshape(y_a[i], (-1, self.config.output_num*self.config.dimension))
        print (np.average(np.linalg.norm(self.g_ar[i],1)), np.average(np.linalg.norm(y,1)))

    return y_a[-1]

  def loss_augmented_map_predict(self, xd, train=False, inf_iter=None, ascent=True):
    yp = self.loss_augmented_soft_predict(xd, train=train, inf_iter=inf_iter, ascent=ascent)
    return np.argmax(yp, 2)

  def train_batch(self, xbatch=None, ybatch=None, verbose=0):
    raise NotImplementedError


  def train_unsupervised_batch(self, xbatch=None, ybatch=None, verbose=0):
    tflearn.is_training(True, self.sess)

    x_b, y_h, y_l, l_h, l_l = self.get_first_large_consecutive_diff(xinput=xbatch, yinput=ybatch, ascent=True)
    bs = np.size(l_h)
    if np.size(l_h) > 1:
      #print (x_b[0:5, 0:20])

      _, o1, n1, v1, v2, e1, e2  = self.sess.run([self.train_step, self.objective, self.num_update, self.vh_sum, self.vl_sum, self.eh_sum, self.el_sum],
              feed_dict={self.x:x_b[:bs, :],
                         self.yp_h_ind:np.reshape(y_h[:bs,:], (-1, self.config.output_num * self.config.dimension)),
                         self.yp_l_ind:np.reshape(y_l[:bs,:], (-1, self.config.output_num * self.config.dimension)),
                         self.value_l: l_l[:bs],
                         self.value_h: l_h[:bs],
                         self.learning_rate_ph:self.config.learning_rate,
                         self.dropout_ph: self.config.dropout,
                         self.margin_weight_ph: self.config.margin_weight})
      if verbose>0:
        print (self.train_iter, o1, n1, v1,v2, e1,e2, np.shape(xbatch)[0], np.shape(x_b)[0])
    else:
      if verbose>0:
        print ("skip")
    return np.size(l_h)


  def train_supervised_batch(self, xbatch, ybatch, verbose=0):
    tflearn.is_training(True, self.sess)

    if self.config.dimension > 1:
      yt_ind = self.var_to_indicator(ybatch)
      yt_ind = np.reshape(yt_ind, (-1, self.config.output_num*self.config.dimension))
    else:
      yt_ind = ybatch

    yp_ind = self.loss_augmented_soft_predict(xinput=xbatch, yinput=yt_ind, train=True, inf_iter=self.config.inf_iter, ascent=True, loss_aug=True)
    yp = np.argmax(yp_ind,-1)
    yp_ind = self.var_to_indicator(yp)
    #yp_ind = self.soft_predict(xinput=xbatch, train=True, ascent=True, inf_iter=self.config.inf_iter)

    yp_ind = np.reshape(yp_ind, (-1, self.config.output_num*self.config.dimension))
    if verbose > 1:
      print (yp_ind[0])

    feeddic = {self.x:xbatch, self.yp: yp_ind, self.yt: yt_ind,
               self.learning_rate_ph:self.config.learning_rate,
               self.margin_weight_ph: self.config.margin_weight,
               self.dropout_ph: self.config.dropout}

    _, o,ce, n, en_yt, en_yhat = self.sess.run([self.train_step, self.objective, self.ce, self.num_update, self.total_energy_yt, self.total_energy_yp], feed_dict=feeddic)
    if verbose > 0:
      print (self.train_iter ,o,n, en_yt, en_yhat, np.average(ce))
    return o

  def train_supervised_e2e_batch(self, xbatch, ybatch, verbose=0):
    tflearn.is_training(True, self.sess)
    if self.config.dimension > 1:
      yt_ind = self.var_to_indicator(ybatch)
      yt_ind = np.reshape(yt_ind, (-1, self.config.output_num * self.config.dimension))
    else:
      yt_ind = ybatch

    yp_init = np.random.normal(0, 1, size=(np.shape(xbatch)[0], self.config.dimension * self.config.output_num))
    feeddic = {self.x: xbatch, self.yt_ind: yt_ind,
               self.yp_ind: yp_init,
               self.learning_rate_ph: self.config.learning_rate,
               self.inf_penalty_weight_ph: self.config.inf_penalty,
               self.dropout_ph: self.config.dropout}

    _, o, ind_ar, en_ar, g_ar, l_ar = self.sess.run([self.train_step, self.objective,self.ind_ar, self.en_ar, self.g_ar, self.l_ar ], feed_dict=feeddic)

    if verbose > 0:
      print("----------------------------------------------------------")
      for i in range(int(self.config.inf_iter)):
        print (g_ar[i],  ind_ar[i], np.average(en_ar[i]), l_ar[i])

    return o

  def save(self, path):
    self.saver.save(self.sess, path)

  def restore(self, path):
    self.saver.restore(self.sess, path)

  def train_supervised_value_batch(self, xbatch, ybatch, verbose=0):
    tflearn.is_training(True, self.sess)
    if self.config.dimension > 1:
      yt_ind = self.var_to_indicator(ybatch)
      yt_ind = np.reshape(yt_ind, (-1, self.config.output_num * self.config.dimension))
    else:
      yt_ind = ybatch

    yp_init = np.random.normal(0, 1, size=(np.shape(xbatch)[0], self.config.dimension * self.config.output_num))
    feeddic = {self.x: xbatch, self.yt_ind: yt_ind,
               self.yp_ind: yp_init,
               self.learning_rate_ph: self.config.learning_rate,
               self.inf_penalty_weight_ph: self.config.inf_penalty,
               self.dropout_ph: self.config.dropout}

    _, o, ind_ar, en_ar, g_ar, l_ar, v_ar, ld_ar = self.sess.run([self.train_step, self.objective,self.ind_ar, self.en_ar, self.g_ar, self.l_ar, self.v_ar, self.ld_ar ], feed_dict=feeddic)

    if verbose > 0:
      print("----------------------------------------------------------")
      if verbose > 1:
        print (v_ar[-1])
      for i in range(int(self.config.inf_iter)):
        print (g_ar[i],  ind_ar[i], np.average(en_ar[i]), l_ar[i], np.average(v_ar[i]), ld_ar[i])

    return o

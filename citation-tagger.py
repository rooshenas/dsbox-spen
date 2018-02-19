
from dsbox.spen.core import spen, config, energy
import argparse
import numpy as np
from dsbox.spen.utils.metrics import token_level_loss_ar, token_level_loss
from dsbox.spen.utils.datasets import get_layers, get_citation_data
import os
import time
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-lr', dest='learning_rate', nargs='?', help='Learning rate')
parser.add_argument('-ir', dest='inf_rate', nargs='?', help='Inference rate')
parser.add_argument('-it', dest='inf_iter', nargs='?', help='Inference iteration')
parser.add_argument('-mw', dest='margin_weight', nargs='?', help='Margin weight')
parser.add_argument('-ln', dest='labeled_num', nargs='?', help='Number of labeled data')
parser.add_argument('-l2', dest='l2_penalty', nargs='?', help='L2 penalty')

args = parser.parse_args()


dataset = 'citation'

if args.labeled_num:
  ln = int(args.labeled_num)
else:
  ln = 1e10

if args.l2_penalty:
    l2 = float(args.l2_penalty)
else:
    l2 = 0.0

if args.learning_rate:
    lr = float(args.learning_rate)
else:
    lr = 0.001

if args.inf_iter:
  inf_iter = float(args.inf_iter)
else:
  inf_iter = 10

if args.inf_rate:
    inf_rate = float(args.inf_rate)
else:
    inf_rate = 0.1

if args.margin_weight:
  mw = float(args.margin_weight)
else:
  mw = 100.0

bs = 100


def perf(ytr_pred, yval_pred, yts_pred, ydata, yval, ytest):
  global best_val
  global test_val

  hm_ts, ex_ts = token_level_loss(yts_pred, ytest)
  hm_tr, ex_tr = token_level_loss(ytr_pred, ydata)
  hm_val, ex_val = token_level_loss(yval_pred, yval)
  if ex_val > best_val:
    best_val = ex_val
    test_val = ex_ts
  print ("Train: %0.3f Val: %0.3f Test: %0.3f ------ Best Val: %0.3f Test: %0.3f" % (
      ex_tr, ex_val, ex_ts, best_val, test_val))

def train_sup(spen, num_steps):
  global ln
  global bs
  global xdata, xtest, xval, ydata, ytest, yval
  global embedding

  ntrain = np.shape(xdata)[0]
  spen.init()
  spen.init_embedding(embedding)

  labeled_num = min((ln, ntrain))
  indices = np.arange(labeled_num)
  #xlabeled = xdata[indices][:]
  #ylabeled = ydata[indices][:]

  xorig = xdata[indices]
  yorig = ydata[indices]

  rotatedx = xorig[:]
  rotatedy = yorig[:]
  for i in range(-50, 50, 1):
    xshift = np.roll(xorig, i, 1)
    yshift = np.roll(yorig, i, 1)
    rotatedx = np.vstack((rotatedx, xshift))
    rotatedy = np.vstack((rotatedy, yshift))


  xlabeled = rotatedx
  ylabeled = rotatedy
  total_num = np.shape(xlabeled)[0]

  for i in range(1, num_steps):
    bs = min((bs, labeled_num))
    perm = np.random.permutation(total_num)

    for b in range(total_num / bs):

      indices = perm[b * bs:(b + 1) * bs]

      xbatch = xlabeled[indices][:]
      ybatch = ylabeled[indices][:]

      spen.set_train_iter(i)
      o = spen.train_batch(xbatch, ybatch, verbose=0)


      print (i, b, o, bs)


    yts_out = spen.map_predict(xtest)
    yval_out = spen.map_predict(xval)
    ytr_out = spen.map_predict(xdata)
    perf(ytr_out, yval_out, yts_out, ydata, yval, ytest)



xdata, xval, xtest, ydata, yval, ytest, x_unlab = get_citation_data()
with open('/iesl/canvas/pedram/CORA/vocab_labels.pickle') as f:
  labeldic = pickle.load(f)

with open('/iesl/canvas/pedram/CORA/vocab_x.pickle') as f:
  worddic = pickle.load(f)

with open('/iesl/canvas/pedram/CORA/trained_embedding.pickle') as f:
  embedding = pickle.load(f)
  vocabulary_size = 20608
  embedding_size = 100

print(np.shape(xdata))

output_num = np.shape(ydata)[1]
input_num = np.shape(xdata)[1]

f_layers, en_layers = get_layers(dataset)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
best_val = 0.0
test_val = 0.0

config = config.Config()
config.l2_penalty = l2
config.inf_iter = inf_iter
config.inf_rate = inf_rate
config.learning_rate = lr
config.dimension = 14
config.output_num = output_num
config.input_num = input_num
config.en_layer_info = en_layers
config.layer_info = f_layers
config.margin_weight = mw
config.output_num = output_num


s = spen.SPEN(config)
e = energy.EnergyModel(config)

#s.eval = lambda xd, yd, yt : token_level_loss_ar(yd, yt)
s.get_energy = e.get_energy_mlp_emb
s.train_batch = s.train_supervised_batch

s.construct_embedding(embedding_size,vocabulary_size)
s.construct(training_type=spen.TrainingType.SSVM)
s.print_vars()

start = time.time()
train_sup(s, 10000)
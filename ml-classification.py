
from dsbox.spen.core import spen, config, energy
import argparse
import numpy as np
from dsbox.spen.utils.metrics import f1_score, hamming_loss
from dsbox.spen.utils.datasets import get_layers, get_data_val
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('-lr', dest='learning_rate', nargs='?', help='Learning rate')
parser.add_argument('-ir', dest='inf_rate', nargs='?', help='Inference rate')
parser.add_argument('-it', dest='inf_iter', nargs='?', help='Inference iteration')
parser.add_argument('-mw', dest='margin_weight', nargs='?', help='Margin weight')
parser.add_argument('-ln', dest='labeled_num', nargs='?', help='Number of labeled data')
parser.add_argument('-l2', dest='l2_penalty', nargs='?', help='L2 penalty')
parser.add_argument('-data', dest='dataset', nargs='?', help='Dataset name')
parser.add_argument('-dp', dest='dropout', nargs='?', help='Dropout')

args = parser.parse_args()


if args.dataset:
  dataset = args.dataset
else:
  dataset = 'bibtex'

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

if args.dropout:
  dp = float(args.dropout)
else:
  dp = 0.0

bs = 100

def perf(ytr_pred, yval_pred, yts_pred, ydata, yval, ytest):
  global best_val_f1
  global test_f1
  ts_f1 = f1_score(yts_pred, ytest)
  tr_f1 = f1_score(ytr_pred, ydata)
  val_f1 = f1_score(yval_pred, yval)
  if val_f1 > best_val_f1:
    best_val_f1 = val_f1
    test_f1 = ts_f1
  hm_ts, ex_ts = hamming_loss(yts_pred, ytest)
  hm_tr, ex_tr = hamming_loss(ytr_pred, ydata)
  hm_val, ex_val = hamming_loss(yval_pred, yval)
  tr_avg = np.average(np.sum(np.round(ytr_pred), 1))
  val_avg = np.average(np.sum(np.round(yval_pred), 1))
  test_avg = np.average(np.sum(np.round(yts_pred), 1))
  print("------- %0.1f %0.1f %0.1f ------ %0.3f %0.3f %0.3f - %0.3f %0.3f %0.3f - %0.3f %0.3f %0.3f ---------------" % (
  tr_avg, val_avg, test_avg, tr_f1, val_f1, ts_f1, hm_tr, hm_val, hm_ts, ex_tr, ex_val, ex_ts))
  print("-- best : %0.4f %0.4f" % (best_val_f1, test_f1))


def train_sup(spen, num_steps):
  global ln
  global bs
  global xdata, xtest, xval, ydata, ytest, yval

  ntrain = np.shape(xdata)[0]
  spen.init()
  mean = np.mean(xdata, axis=0).reshape((1, -1))
  std = np.std(xdata, axis=0).reshape((1, -1)) + 1e-20
  xdata -= mean
  xdata /= std

  xtest -= mean
  xtest /= std
  xval -= mean
  xval /= std

  labeled_num = min((ln, ntrain))
  indices = np.arange(labeled_num)
  xlabeled = xdata[indices][:]
  ylabeled = ydata[indices][:]

  total_num = np.shape(xlabeled)[0]

  for i in range(1, num_steps):
    bs = min((bs, labeled_num))
    perm = np.random.permutation(total_num)

    for b in range(ntrain / bs):

      indices = perm[b * bs:(b + 1) * bs]

      xbatch = xlabeled[indices][:]
      ybatch = ylabeled[indices][:]

      spen.set_train_iter(i)
      o = spen.train_batch(xbatch, ybatch)
      #print (i, b, o)


    if i % 2 == 0:
      yts_out = spen.map_predict(xinput=xtest)
      yval_out = spen.map_predict(xinput=xval)
      ytr_out = spen.map_predict(xinput=xdata)
      perf(ytr_out, yval_out, yts_out, ydata, yval, ytest)



xdata, xval, xtest, ydata, yval, ytest= get_data_val(dataset,0.3)

print(np.shape(xdata))

output_num = np.shape(ydata)[1]
input_num = np.shape(xdata)[1]

f_layers, en_layers = get_layers(dataset)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
best_val_f1 = 0.0
test_f1 = 0.0

config = config.Config()
config.l2_penalty = l2
config.inf_iter = inf_iter
config.inf_rate = inf_rate
config.learning_rate = lr
config.dropout = dp
config.dimension = 2
config.output_num = output_num
config.input_num = input_num
config.en_layer_info = en_layers
config.layer_info = f_layers
config.margin_weight = mw
config.output_num = output_num

s = spen.SPEN(config)
e = energy.EnergyModel(config)

#s.eval = lambda xd, yd, yt : f1_score_c_ar(yd, yt)
s.get_energy = e.get_energy_mlp
#s.prediction_net = e.softmax_prediction_network
s.train_batch = s.train_supervised_batch

s.construct(training_type=spen.TrainingType.SSVM)
s.print_vars()

start = time.time()
train_sup(s, 10000)

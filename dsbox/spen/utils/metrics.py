
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

def token_level_loss(cpred, ctrue):
  try:
    label_num = np.shape(cpred)[1]
    pred = cpred.astype(np.int)
    true = ctrue.astype(np.int)
    n1 = len(pred)
    n2 = len(true)
    assert n1 == n2
    hloss = 0.0
    exact_acc = 0.0
    for i in range(n1):
      sample_loss = 0.0
      exact = 0.0
      xp = pred[i]
      xt = true[i]
      cnp = 0
      for j in range(label_num):
        if (xt[j] == 0):
          continue
        cnp+=1
        if xp[j] != xt[j]:
          sample_loss += 1.0
        else:
          exact +=1
      sample_loss = sample_loss / cnp
      exact = exact / cnp
      hloss += sample_loss
      exact_acc += exact
    return (hloss / n1, exact_acc / n1)
  except Warning:
    return (0.0,0.0);

def token_level_loss_ar(cpred, ctrue ):
  label_num = np.shape(cpred)[1]
  results = []
  try:
    pred = cpred.astype(np.int)
    true = ctrue.astype(np.int)
    n1 = len(pred)
    n2 = len(true)
    assert n1 == n2
    hloss = 0.0
    exact_acc = 0.0
    for i in range(n1):
      sample_loss = 0.0
      exact = 0.0
      xp = pred[i]
      xt = true[i]
      cnp = 0.0
      for j in range(label_num):
        if (xt[j] == 0):
          continue
        cnp+=1.0
        if xp[j] != xt[j]:
          sample_loss += 1.0
        else:
          exact += 1.0
      sample_loss = sample_loss / cnp
      exact = exact / cnp
      results.append(exact)
      hloss += sample_loss
      exact_acc += exact
    return results
  except Warning:
    return None




def hamming_loss(cpred, ctrue, cut=0.5):
  try:
    label_num = np.shape(cpred)[1]
    pred = (cpred >= cut).astype(np.int)
    true = ctrue.astype(np.int)
    n1 = len(pred)
    n2 = len(true)
    assert n1 == n2
    hloss = 0.0
    exact = 0.0
    for i in range(n1):
      sample_loss = 0
      xp = pred[i]
      xt = true[i]
      for j in range(label_num):
        if xp[j] != xt[j]:
          sample_loss += 1.0
      if sample_loss <= 0.0:
        exact += 1
      sample_loss = sample_loss / label_num
      hloss += sample_loss
    return (hloss / n1, exact / n1)
  except Warning:
    return (0.0,0.0);

def h_score_ar(cpred, ctrue, cut=0.5):
  label_num = np.shape(cpred)[1]
  results = []
  pred = (cpred >= cut).astype(np.int)
  true = ctrue.astype(np.int)
  n1 = len(pred)
  n2 = len(true)
  assert n1 == n2
  for i in range(n1):
    sample_loss = 0
    xp = pred[i]
    xt = true[i]
    for j in range(label_num):
      if xp[j] != xt[j]:
        sample_loss += 1.0
    sample_loss = sample_loss / label_num
    results.append(1.0 - sample_loss)
  return results

def ex_ar(cpred, ctrue, cut=0.5):
  label_num = np.shape(cpred)[1]
  results = []
  pred = (cpred >= cut).astype(np.int)
  true = ctrue.astype(np.int)
  n1 = len(pred)
  n2 = len(true)
  assert n1 == n2
  for i in range(n1):
    sample_loss = 0
    xp = pred[i]
    xt = true[i]
    for j in range(label_num):
      if xp[j] != xt[j]:
        sample_loss += 1.0
    results.append(0.0) if sample_loss > 0  else results.append(1.0)
  return results

def f1_score(cpred, ctrue, cut=0.5):
  try:
    label_num = np.shape(cpred)[1]
    pred = (cpred >= cut).astype(np.int)
    true = ctrue.astype(np.int)
    n1 = len(pred)
    n2 = len(true)
    assert n1 == n2
    f1_score = 0.0
    for i in range(n1):
      sample_loss = 0
      intersection = 0.0
      xp = pred[i]
      xt = true[i]
      for j in range(label_num):
        if xp[j] != xt[j]:
          sample_loss += 1.0
        else:
          intersection += xt[j]
      if (np.sum(xt) + np.sum(xp)) > 0 :
        f1_score += (2.0*intersection) / (np.sum(xp) + np.sum(xt))
      else:
        return None
    return f1_score / n1
  except Warning:
    return 0.0


def f1_macro(cpred, ctrue,cut=0.5):
  pred = (cpred >= cut).astype(np.float32)
  true = ctrue.astype(np.float32)
  n1 = len(pred)
  n2 = len(true)
  assert n1 == n2
  np.sum(pred, 0)

  intersection =np.sum(np.multiply(pred,true))

  alltrue = np.sum(true)
  allpred = np.sum(pred)
  if alltrue == 0:
    return 0.0
  if allpred == 0:
    raise ArithmeticError


  precision = intersection / allpred
  recall = intersection/ alltrue

  return (2.0*precision*recall) / (precision+recall), precision, recall


def h_score_c_ar(cpred, ctrue, num_labels):
  min = np.log(np.minimum(cpred, ctrue) + 1e-10)
  max = np.log(np.maximum(cpred, ctrue) + 1e-10)
  return 1.0-(np.sum(np.exp(min - max),1)/num_labels)



def f1_score_c_ar(cpred, ctrue):
  intersection = np.sum(np.minimum(cpred,ctrue),1)
  union = np.sum(np.maximum(cpred,ctrue),1)
  return np.divide(2.0*intersection, union + intersection)



def precision_c_ar(cpred, ctrue):
  intersection = np.sum(np.minimum(cpred,ctrue),1)
  prediction = np.sum(cpred,1)
  return np.divide(intersection, prediction)

def recall_c_ar(cpred, ctrue):
  intersection = np.sum(np.minimum(cpred,ctrue),1)
  true = np.sum(ctrue,1)
  return np.divide(intersection, true)

def cross_entropy_ar(cpred,ctrue):
  log_cpred = np.log(cpred + 1e-10)
  return np.sum(-np.multiply(ctrue, log_cpred) - np.multiply(1-ctrue, 1-log_cpred),1)



def f1_loss_ar(cpred, ctrue, label_num, cut=0.5):
  try:
    results = []
    pred = (cpred >= cut).astype(np.int)
    true = ctrue.astype(np.int)
    n1 = len(pred)
    n2 = len(true)
    assert n1 == n2
    f1_score = 0.0
    for i in range(n1):
      sample_loss = 0
      intersection = 0.0
      xp = pred[i]
      xt = true[i]
      for j in range(label_num):
        if xp[j] != xt[j]:
          sample_loss += 1.0
        else:
          intersection += xt[j]
      if np.sum(xt) > 0 :
        results.append ( 1 -  (2.0*intersection) / (np.sum(xp) + np.sum(xt)))
      else:
        results.append(0.0)
    return results
  except Warning:
    return None

def f1_score_ar(cpred, ctrue, label_num, cut=0.5):
  results = []
  pred = (cpred >= cut).astype(np.int)
  true = ctrue.astype(np.int)
  n1 = len(pred)
  n2 = len(true)
  assert n1 == n2
  for i in range(n1):
    sample_loss = 0
    intersection = 0.0
    xp = pred[i]
    xt = true[i]
    for j in range(label_num):
      if xp[j] != xt[j]:
        sample_loss += 1.0
      else:
        intersection += xt[j]
    if np.sum(xt) > 0:
      results.append((2.0 * intersection) / (np.sum(xp) + np.sum(xt)))
    else:
      results.append(1.0)
  return results


def mutual_info(x, y):
  bins = 2
  #c_xy = np.histogram2d(x, y, bins)[0]
  mi = normalized_mutual_info_score(x, y)
  return mi


if __name__ == "__main__":
  a = np.round(np.random.uniform(0.5,1, (100,20)))
  b = np.round(np.random.uniform(0.6,1,(100,20)))
  #print np.shape(f1_score_c_ar(a,b))
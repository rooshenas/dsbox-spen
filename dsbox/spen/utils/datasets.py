import numpy as np
import pickle
import pandas as pd


def get_layers(dataset):
  yeastlayers = [(300, 'relu'), (200, 'relu')]
  yeastenlayers = [(100, 'relu'), (50, 'relu')]


  scene_layers = [(300, 'relu'), (200, 'relu')]
  scene_enlayers = [(20, 'relu'), (20, 'relu')]

  bibtexlayers = [(1000, 'relu')]
  bibtex_enlayers = [(250, 'softplus')]

  dellayers = [(2500, 'relu'), (1500, 'relu')]
  # del_enlayers = [(100,'relu'), (50, 'relu')]
  del_enlayers = [(500, 'relu'), (300, 'relu')]

  bookmarks_layers = [(1500, 'relu'), (1000, 'relu')]
  bookmarks_enlayers = [(500, 'relu'), (200, 'relu')]


  ppi_layers = [(500, 'relu'), (200, 'relu')]
  ppi_enlayers = [(100, 'relu'), (50, 'relu')]

  nltcs_layers = [(500, 'relu'), (300, 'relu')]
  nltcs_enlayers = [(100, 'relu'), (50, 'relu')]

  plants_layers = [(500, 'relu'), (300, 'relu')]
  plants_enlayers = [(200, 'relu'), (100, 'relu')]

  dna_layers = [(100, 'relu'), (100, 'relu')]
  dna_enlayers = [(100, 'relu'), (100, 'relu')]

  jester_layers = [(200, 'relu'), (100, 'relu')]
  jester_enlayers = [(200, 'relu'), (100, 'relu')]

  cwebkb_layers = [(500, 'relu'), (300, 'relu')]
  cwebkb_enlayers = [(300, 'relu'), (200, 'relu')]

  mnist_layers = [(1000, 'relu'), (500, 'relu')]
  mnist_enlayers = [(200, 'relu'), (100, 'relu')]


  c20ng_layers = [(1000, 'relu'), (500, 'relu')]
  c20ng_enlayers = [(300, 'relu'),(200, 'relu')]


  citation_layers = [(1000, 'relu'),(500, 'relu')]
  citation_enlayers = [(1000, 'softplus')]

  medical_layers = [(1000, 'relu'), (500, 'relu')]
  medical_enlayers = [(100, 'softplus')]

  kddcup_layers = [(1500, 'relu'), (1500, 'relu')]
  kddcup_enlayers = [(500, 'relu'), (500, 'relu')]



  datasets = {"scene":[],
            "yeast":(yeastlayers,yeastenlayers),
            "medical":(medical_layers, medical_enlayers),
            "bibtex":(bibtexlayers, bibtex_enlayers),
            "bibtex_short":(bibtexlayers, bibtex_enlayers),
            'delicious':(dellayers, del_enlayers),
            "bookmarks": (bookmarks_layers, bookmarks_enlayers),
            "scene": (scene_layers, scene_enlayers),
            "mnist": (mnist_layers, mnist_enlayers),
            "nltcs": (nltcs_layers, nltcs_enlayers),
            "plants": (plants_layers, plants_enlayers),
            "dna": (dna_layers, dna_enlayers),
            "jester": (jester_layers, jester_enlayers),
            "cwebkb": (cwebkb_layers, cwebkb_enlayers),
            "ppi": (ppi_layers, ppi_enlayers),
            "c20ng": (c20ng_layers, c20ng_enlayers),
            "citation": (citation_layers, citation_enlayers),
            "medical": (medical_layers, medical_enlayers),
            "kddcup": (kddcup_layers, kddcup_enlayers)}

  if not dataset in datasets:
    return (plants_layers, plants_enlayers)
  return datasets[dataset]


def load_data(file):
  data = []
  for l in open(file, 'r'):
    y=map(lambda x: -1 if x == "*" else float(x), l.strip().split(','))
    data.append(list(y))
  data = np.array(data)
  return data

def get_data(dataset):
  xtrfile = "/iesl/canvas/pedram/mlcond/%s.ts.ev" % dataset
  xtsfile = "/iesl/canvas/pedram/mlcond/%s.test.ev" % dataset
  ytrfile = "/iesl/canvas/pedram/mlcond/%s.ts.data" % dataset
  ytsfile = "/iesl/canvas/pedram/mlcond/%s.test.data" % dataset

  ytest = load_data(ytsfile)
  ydata = load_data(ytrfile)
  xtest = load_data(xtsfile)
  xdata = load_data(xtrfile)
  return (xdata,xtest,ydata,ytest)

def get_data_y(dataset):
  ytrfile = "/iesl/canvas/pedram/mlcond/%s.ts.data" % dataset
  ytsfile = "/iesl/canvas/pedram/mlcond/%s.test.data" % dataset

  ytest = load_data(ytsfile)
  ydata = load_data(ytrfile)
  return (ydata,ytest)

def get_mnist():
  xtrfile = "/iesl/canvas/pedram/mnist/mnist-2q.ts.ev"
  xtsfile = "/iesl/canvas/pedram/mnist/mnist-2q.test.ev"
  xvalfile = "/iesl/canvas/pedram/mnist/mnist-2q.valid.ev"
  ytrfile = "/iesl/canvas/pedram/mnist/mnist-2q.ts.data"
  ytsfile = "/iesl/canvas/pedram/mnist/mnist-2q.test.data"
  yvalfile = "/iesl/canvas/pedram/mnist/mnist-2q.valid.data"

  xdata = load_data(xtrfile)
  xval = load_data(xvalfile)
  xtest = load_data(xtsfile)

  ydata = load_data(ytrfile)
  yval = load_data(yvalfile)
  ytest = load_data(ytsfile)

  return xdata, xval, xtest, ydata, yval, ytest

def get_fashion():
  xtrfile = "/iesl/canvas/pedram/fashion/fashion-2q.ts.ev"
  xtsfile = "/iesl/canvas/pedram/fashion/fashion-2q.test.ev"
  xvalfile ="/iesl/canvas/pedram/fashion/fashion-2q.valid.ev"
  ytrfile = "/iesl/canvas/pedram/fashion/fashion-2q.ts.data"
  ytsfile = "/iesl/canvas/pedram/fashion/fashion-2q.test.data"
  yvalfile ="/iesl/canvas/pedram/fashion/fashion-2q.valid.data"

  xdata = load_data(xtrfile)
  xval = load_data(xvalfile)
  xtest = load_data(xtsfile)

  ydata = load_data(ytrfile)
  yval = load_data(yvalfile)
  ytest = load_data(ytsfile)

  return xdata, xval, xtest, ydata, yval, ytest


def get_7scene(crop=None):
  path = '/iesl/canvas/jburroni/proximalDeepStructure/train/'
  def load_image(name):
    data_buffer = np.fromfile(path + name, dtype=np.int16)
    img = data_buffer[2:]
    img = img.reshape((data_buffer[0], data_buffer[1])).astype(float)
    return np.clip(img / 5000, 0, 1)


  def _load_dataset_from_list(list_name, path, crop=None):
    list = pd.read_csv(path + list_name, index_col=0, names=['noisy', 'gt'], sep=' ').drop_duplicates()
    elements = []

    def accumulator(serie):
      noisy, gt = load_image(serie[0]), load_image(serie[1])
      if crop:
        noisy, gt = crop(noisy, gt)
      elements.append((noisy, gt))

    list.apply(accumulator, axis=1)
    x, y = zip(*elements)
    return np.stack(x), np.stack(y)


  f = lambda x: list(_load_dataset_from_list(x, path, crop))
  xtrain, ytrain = f('train.lst')
  xval, yval = f('val.lst')
  xlength = np.shape(xtrain)[1]
  ylength = np.shape(xtrain)[2]

  return np.reshape(xtrain, (-1, xlength*ylength)), np.reshape(xval, (-1, xlength*ylength)),np.reshape(ytrain, (-1, xlength*ylength)),np.reshape(yval, (-1, xlength*ylength))

def get_ppi(ratio):
  ffile = "/iesl/canvas/pedram/net/ppi-features-mpe.txt"
  lfile = "/iesl/canvas/pedram/net/ppi-groups.txt"
  allxdata = load_data(ffile)
  allydata = load_data(lfile)
  size = np.shape(allxdata)[0]
  indices =range(size)
  np.random.shuffle(indices)
  val_ratio = ratio
  test_ratio = 0.5

  test_num = int(size*test_ratio)
  val_num = int(size*(1-test_ratio)*val_ratio)
  test_indices = indices[0:test_num]
  val_indices = indices[test_num:test_num+val_num]
  train_indices = indices[test_num + val_num:]

  xdata = allxdata[train_indices]
  ydata = allydata[train_indices]
  xval = allxdata[val_indices]
  yval = allydata[val_indices]
  xtest = allxdata[test_indices]
  ytest = allydata[test_indices]


  return (xdata,xval, xtest,ydata,yval, ytest)

def get_citation_data():
    with open('/iesl/canvas/pedram/CORA/X_train.pickle') as f:
      xdata = pickle.load(f)

    with open('/iesl/canvas/pedram/CORA/Y_train.pickle') as f:
      ydata = pickle.load(f)

    with open('/iesl/canvas/pedram/CORA/X_dev.pickle') as f:
      xval = pickle.load(f)

    with open('/iesl/canvas/pedram/CORA/Y_dev.pickle') as f:
      yval = pickle.load(f)

    with open('/iesl/canvas/pedram/CORA/X_test.pickle') as f:
      xtest = pickle.load(f)

    with open('/iesl/canvas/pedram/CORA/Y_test.pickle') as f:
      ytest = pickle.load(f)

    with open('/iesl/canvas/pedram/CORA/X_unlabelled.pickle') as f:
      x_unlab = pickle.load(f)

    return xdata, xval, xtest, ydata, yval, ytest, x_unlab

def get_data_lim(dataset, evnum, size):
  trfile = "/iesl/canvas/pedram/%d_data/%s.ts.data" % (size,dataset)
  valfile = "/iesl/canvas/pedram/data/%s.valid.data" % dataset
  tsfile = "/iesl/canvas/pedram/data/%s.test.data" % dataset

  evtrfile = "/iesl/canvas/pedram/%d_ev%d/%s.ts.ev" % (size,evnum, dataset)
  evvalidfile = "/iesl/canvas/pedram/ev%d/%s.valid.ev" % (evnum, dataset)
  evtsfile = "/iesl/canvas/pedram/ev%d/%s.test.ev" % (evnum, dataset)

  test = load_data(tsfile)
  data = load_data(trfile)

  valid = load_data(valfile)

  ev_test = load_data(evtsfile)
  ev_data =load_data(evtrfile)
  ev_val = load_data(evvalidfile)

  ex = ev_data[0]
  evvars = []
  yvars = []
  for i in range(0, len(ex)):
    if ex[i] < 0:
      yvars.append(i)
    else:
      evvars.append(i)

  xdata = data[:, evvars]
  xval  = valid[:, evvars]
  xtest = test[:, evvars]

  ydata = data[:,yvars]
  yval  = valid[:, yvars]
  ytest = test[:, yvars]

  return (xdata,xval, xtest,ydata,yval, ytest, evvars, yvars)


def get_medical():
  dataset = 'medical'
  xtrfile = "/iesl/canvas/pedram/medical/%s.ts.ev" % dataset
  xtsfile = "/iesl/canvas/pedram/medical/%s.test.ev" % dataset
  ytrfile = "/iesl/canvas/pedram/medical/%s.ts.data" % dataset
  ytsfile = "/iesl/canvas/pedram/medical/%s.test.data" % dataset

  ytest = load_data(ytsfile)
  ydataval = load_data(ytrfile)
  xtest = load_data(xtsfile)
  xdataval = load_data(xtrfile)

  size = np.shape(xdataval)[0]
  indices = range(size)
  np.random.shuffle(indices)
  val_num = int(size * 0.3)
  val_indices = indices[0:val_num]
  train_indices = indices[val_num:]

  xval = xdataval[val_indices, :]
  yval = ydataval[val_indices, :]
  xdata = xdataval[train_indices, :]
  ydata = ydataval[train_indices, :]

  yvars = range(45)
  evvars = range(45,45+1449)
  return (xdata, xval, xtest,ydata, yval, ytest, evvars, yvars)

def get_20ng():
  trfile = '/iesl/canvas/pedram/20ng/20newsgroup-2k.train.data'
  tsfile = '/iesl/canvas/pedram/20ng/20newsgroup-2k.test.data'
  evtrfile = '/iesl/canvas/pedram/20ng/20newsgroup-2k.train.ev'
  evtsfile = '/iesl/canvas/pedram/20ng/20newsgroup-2k.test.ev'

  test = load_data(tsfile)
  data = load_data(trfile)
  #valid = load_data(valfile)


  ev_test = load_data(evtsfile)
  ev_data =load_data(evtrfile)
  #ev_val = load_data(evvalidfile)


  ex = ev_data[0]
  evvars = []
  yvars = []
  for i in range(0, len(ex)):
    if ex[i] < 0:
      yvars.append(i)
    else:
      evvars.append(i)

  xdataval = data[:, evvars]
  #xval  = valid[:, evvars]
  xtest = test[:, evvars]

  ydataval = data[:,yvars]
  #yval  = valid[:, yvars]
  ytest = test[:, yvars]

  size = np.shape(xdataval)[0]
  indices = range(size)
  np.random.shuffle(indices)
  val_num = int(size * 0.3)
  val_indices = indices[0:val_num]
  train_indices = indices[val_num:]

  xval = xdataval[val_indices, :]
  yval = ydataval[val_indices, :]
  xdata = xdataval[train_indices, :]
  ydata = ydataval[train_indices, :]

  return (xdata, xval, xtest,ydata, yval, ytest, evvars, yvars)

def get_data_libra(dataset, evnum):
  trfile = "/iesl/canvas/pedram/data/%s.ts.data" % dataset
  valfile = "/iesl/canvas/pedram/data/%s.valid.data" % dataset
  tsfile = "/iesl/canvas/pedram/data/%s.test.data" % dataset

  evtrfile = "/iesl/canvas/pedram/ev%d/%s.ts.ev" % (evnum, dataset)
  evvalidfile = "/iesl/canvas/pedram/ev%d/%s.valid.ev" % (evnum, dataset)
  evtsfile = "/iesl/canvas/pedram/ev%d/%s.test.ev" % (evnum, dataset)

  test = load_data(tsfile)
  data = load_data(trfile)
  valid = load_data(valfile)


  ev_test = load_data(evtsfile)
  ev_data =load_data(evtrfile)
  ev_val = load_data(evvalidfile)

  ex = ev_data[0]
  evvars = []
  yvars = []
  for i in range(0, len(ex)):
    if ex[i] < 0:
      yvars.append(i)
    else:
      evvars.append(i)

  xdata = data[:, evvars]
  xval  = valid[:, evvars]
  xtest = test[:, evvars]

  ydata = data[:,yvars]
  yval  = valid[:, yvars]
  ytest = test[:, yvars]

  return (xdata,xval, xtest,ydata,yval, ytest, evvars, yvars)

def get_dependency_data():
  root_path = "/iesl/canvas/svenkitachal/dep_parse/data/"
  with open(root_path + 'X_train.pkl') as f:
    xdata = pickle.load(f)

  with open(root_path + 'pos_train.pkl') as f:
    posdata = pickle.load(f)

  with open(root_path + 'Y_train.pkl') as f:
    ydata = pickle.load(f)

  with open(root_path + 'X_dev.pkl') as f:
    posval = pickle.load(f)

  with open(root_path + 'pos_dev.pkl') as f:
    xval = pickle.load(f)

  with open(root_path + 'Y_dev.pkl') as f:
    yval = pickle.load(f)

  with open(root_path + 'X_test.pkl') as f:
    xtest = pickle.load(f)

  with open(root_path + 'pos_test.pkl') as f:
    postest = pickle.load(f)

  with open(root_path + 'Y_test.pkl') as f:
    ytest = pickle.load(f)

  return xdata, xval, xtest, posdata, posval, postest, ydata, yval, ytest


def get_data_val(dataset, ratio):
  if dataset == "ppi":
    return get_ppi(ratio)
  xtrfile = "/iesl/canvas/pedram/mlcond/%s.ts.ev" % dataset
  xtsfile = "/iesl/canvas/pedram/mlcond/%s.test.ev" % dataset
  ytrfile = "/iesl/canvas/pedram/mlcond/%s.ts.data" % dataset
  ytsfile = "/iesl/canvas/pedram/mlcond/%s.test.data" % dataset

  ytest = load_data(ytsfile)
  ydataval = load_data(ytrfile)
  xtest = load_data(xtsfile)
  xdataval = load_data(xtrfile)
  print(np.shape(xdataval))

  size = np.shape(xdataval)[0]
  indices = np.random.permutation(np.arange(size))
  val_num = int(size*ratio)
  val_indices = indices[0:val_num]
  train_indices = indices[val_num:]
  xval = xdataval[val_indices,:]
  yval = ydataval[val_indices,:]
  xdata = xdataval[train_indices,:]
  ydata = ydataval[train_indices,:]
  return (xdata,xval, xtest,ydata,yval, ytest)

































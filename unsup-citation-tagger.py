
from dsbox.spen.core import ispen as sp, config, energy
import argparse
import numpy as np
from dsbox.spen.utils.metrics import token_level_loss_ar, token_level_loss
from dsbox.spen.utils.datasets import get_layers, get_citation_data
import os
import time
import pickle
import re
import string

parser = argparse.ArgumentParser()
parser.add_argument('-lr', dest='learning_rate', nargs='?', help='Learning rate')
parser.add_argument('-ir', dest='inf_rate', nargs='?', help='Inference rate')
parser.add_argument('-it', dest='inf_iter', nargs='?', help='Inference iteration')
parser.add_argument('-mw', dest='margin_weight', nargs='?', help='Margin weight')
parser.add_argument('-ln', dest='labeled_num', nargs='?', help='Number of labeled data')
parser.add_argument('-l2', dest='l2_penalty', nargs='?', help='L2 penalty')
parser.add_argument('-ip', dest='inf_l2_penalty', nargs='?', help='Inf L2 penalty')

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

if args.inf_l2_penalty:
  ip = float(args.inf_l2_penalty)
else:
  ip = 0.0

bs = 100
debug = False



xdata, xval, xtest, ydata, yval, ytest, x_unlab = get_citation_data()
with open('/iesl/canvas/pedram/CORA/vocab_labels.pickle') as f:
  labeldic = pickle.load(f)

with open('/iesl/canvas/pedram/CORA/vocab_x.pickle') as f:
  worddic = pickle.load(f)

with open('/iesl/canvas/pedram/CORA/trained_embedding.pickle') as f:
  embedding = pickle.load(f)
  vocabulary_size = 20608
  embedding_size = 100

  with open("/iesl/canvas/pedram/CORA/authors.pickle", 'rb') as f:
    author_list = pickle.load(f)

with open("/iesl/canvas/pedram/CORA/journals.pickle", 'rb') as f:
  journal_list = pickle.load(f)

with open("/iesl/canvas/pedram/CORA/conference.pickle", 'rb') as f:
  conf_list = pickle.load(f)

with open("/iesl/canvas/pedram/CORA/locations.pickle", "rb") as f:
  loc_dic = pickle.load(f)

with open("/iesl/canvas/pedram/CORA/titles.pickle", 'rb') as f:
  title_dic = pickle.load(f)

with open("/iesl/canvas/pedram/CORA/universities.pickle", 'rb') as f:
  uni_dic = pickle.load(f)
uni_dic.keys().remove("-")

with open("/iesl/canvas/pedram/CORA/authors_all.pickle", 'rb') as f:
  auth_dic = pickle.load(f)

a_list = dict([(author_list[i], i) for i, x in enumerate(author_list)])
for x in string.ascii_uppercase:
  a_list[x] = 1

j_list = dict([(journal_list[i], i) for i, x in enumerate(journal_list)])

month_list = ["January", "Jan", "February", "Feb", "March", "Mar", "June", "Jun", "July", "Jul", "August", "September",
              "Sept", "October", "Oct", "November", "Nov", "December", "Dec"]
month_dic = dict([(x, i) for i, x in enumerate(month_list)])
year = re.compile("19\d{2}|20\d{2}")

aind = labeldic.index("author")
eind = labeldic.index("editor")
tind = labeldic.index("title")
techind = labeldic.index("tech")
bind = labeldic.index("booktitle")
lind = labeldic.index("location")
jind = labeldic.index("journal")
pind = labeldic.index("pages")
vind = labeldic.index("volume")
pubind = labeldic.index("publisher")
instind = labeldic.index("institution")
locind = labeldic.index("location")
nind = labeldic.index("note")
lpad = labeldic.index("PAD")
wpad = worddic.index("PAD")
wpind = worddic.index("pages")
wjind = worddic.index("Journal")
dind = labeldic.index("date")
wmind  = [worddic.index(m) for m in ["January", "Jan", "February",  "Feb", "March", "Mar", "April",  "Apr", "June", "Jun", "July", "Jul", "August", "Aug", "September", "Sep", "October", "Oct", "November", "Nov", "December", "Dec" ]]
year = re.compile("19\d{2}|20\d{2}")
wlocind = [worddic.index(l) for l in ["CA", "MA", "OR", "TX", "NY", "VA", "IL", "MD", "NJ", "PA", "WA", "VT", "WI", "GA","Zurich", "Munich", "London", "Cambridge", "Oxford", "Sydney", "Melbourne", "USA", "Canada", "England", "Germany", "France","Canada", "Dallas", "Houston", "Atlanta"]]






def evaluate_citation2(xinput=None, yinput=None):
  global debug

  xd=xinput
  yd=yinput
  size = np.shape(xd)[0]

  reward = np.zeros(size)
  penalty = np.zeros(size)


  for i in range(size):
      seen = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1, 12: -1, 13: -1}
      last = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1, 12: -1, 13: -1}
      block = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1, 12: -1, 13: -1}
      block_x = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[]}
      e = yd[i,:]
      x = xd[i,:]
      last_l = 0
      xpad = 1000
      j = 0

      quote_open=False
      parenthesis_open = False
      dash = False
      last_wl = ""
      while (j < len(e) and x[j] != wpad):


        if x[j] in wlocind and e[j] != locind:
          penalty[i] += 10
          if debug: print "41.1"
          #print worddic[x[j]], labeldic[e[j]]


        if x[j] in wmind:
          if e[j] != dind:
            penalty[i] += 10
            if debug: print "41.2"
          #else:
          #  print worddic[x[j]], labeldic[e[j]]

        word = worddic[x[j]]
        wl = word.lower()

        if e[j] == vind or e[j] == dind or e[j] == pind or e[j] == tind:
          block_x[e[j]].append(wl)

        if last_wl == '(' or wl == ')':
          if e[j] != e[j-1]:
            penalty[i] += 1
            if debug: print "41.3"


        if wl == '-' and last_wl == '-':
          if e[j] != pind:
            penalty[i] += 10
            if debug: print "41.4"

        if wl == '.':
          if j > 0 and e[j] != e[j-1]:
            penalty[i] += 1
            if debug: print "20.0"

          #if (len(last_wl) == 1 and ord(last_wl) > ord('a') and ord(last_wl) < ord('z')) or (last_wl == 'pp' or last_wl == 'proc') :
            #print wl
          #  if j> 0 and j < output_num - 2 and e[j] != e[j+1]:
          #    penalty[i] += 1
          #    if debug: print "20.1"
              #print last_wl, labeldic[e[j-1]], labeldic[e[j]], labeldic[e[j+1]]
          #elif j < output_num - 2 and e[j] == e[j+1]:
          #  penalty[i] += 1
          #  if debug: print "20.2"
          #if j > 0 and j < output_num - 2:
          #  if e[j-1] == aind and e[j+1] == aind and e[j] != aind:
          #    penalty[i] += 1
          #    if debug: print "21"


        elif wl == '"' and e[j] != tind:
          penalty[i] += 10
          if debug: print "25"

        elif wl == ',':
          if e[j-1] == aind and e[j+1] == aind and e[j] != aind:
            penalty[i] += 1
            if debug: print "26"


        #Colon only comes in Title
        #elif wl == ':':
        #  if e[j] != tind:
        #    penalty[i] += 10
        #    if debug: print "27.1"
        #  if e[j] != e[j+1]:
        #    penalty[i] += 1
        #    if debug: print "27.2"
        #  if e[j-1] != e[j]:
        #    penalty[i] += 1
        #    if debug: print "27.3"
        #  if e[j-1] != e[j+1]:
        #    penalty[i] += 1
        #    if debug: print "27.4"



        elif wl.startswith('page') or wl == "pp":
          if e[j] != pind:
            penalty[i] += 10
            if debug: print "22"

        #add of, in,
        elif wl == "and" or wl =='-':
          if e[j-1] != e[j]:
            penalty[i] += 1
            if debug: print "23"
          if e[j] != e[j+1]:
            penalty[i] += 1
            if debug: print "24"





        #'The' is only comes in title, booktitle, or journal.
        #elif wl == "the":
        #  if e[j] != tind and e[j] != bind and e[j] != jind:
        #    penalty[i] += 1
        #    if debug: print "24.1"


        #elif wl == "for":
        #  if e[j] != tind:
        #    penalty[i] += 10
        #    if debug: print "24.2"

        elif wl.startswith("novel") or wl.startswith("method") \
            or wl.startswith("approach") or wl.startswith("technique") or wl.startswith("with")\
            or wl.startswith("propert") or wl.startswith("efficient"):
          if e[j] != tind:
            penalty[i] += 10
            if debug: print "24.3"

        elif (wl == "ijcai") and e[j] != bind:
          penalty[i] += 10
          if debug: print "24.4"

        elif wl =="ieee" or wl == "acm" or wl == "siam":
          if e[j] != bind and e[j] != jind:
            penalty[i] += 10
            if debug: print "27"

        elif wl.startswith("proceeding") or wl.startswith("conference") or wl == "proc":
          if e[j] != bind:
            penalty[i] += 10
            if debug: print "28"

        elif (wl.startswith("journal")):
          if e[j] != jind:
            penalty[i] += 10
            if debug: print "28.1"

        elif (wl.startswith("editor") or wl == "ed" ) and e[j] != eind:
          penalty[i] += 10
          if debug: print "29"

        elif (wl.startswith("techninal") or wl.startswith("preprint"))and e[j] != techind:
          penalty[i] += 10
          if debug: print "31"

        elif wl.startswith("date") and e[j] != dind:
          penalty[i] += 10
          if debug: print "32"

        elif (wl == "vol" or wl == "volume") and e[j] != vind:
          penalty[i] += 10
          if debug: print "32.1"


        elif (wl.startswith("university") or wl == "univ" or wl == "institute" ) and e[j] != instind:
          penalty[i] += 10
          if debug: print "33"

        #elif wl.startswith("note") and e[j] != nind:
          #penalty[i] += 10
        #  if debug: print "34"

        elif (len(wl) == 4 and year.search(wl) is not None) and e[j] != dind:
          penalty[i] += 10
          #print wl
          if debug: print wl
          if debug: print labeldic[e[j]]
          if debug: print "35"


        #if j > 0 and j < output_num - 2 and e[j-1] != e[j+1]:
        #  if wl != '.' and wl != ',' and wl != '"' and wl != ')' and wl != '(':
        #    penalty[i] += 1
            #print wl, labeldic[e[j-1]], labeldic[e[j+1]]


        last[e[j]] = j
        if seen[e[j]] < 0:
          seen[e[j]] = j
          block[e[j]] = 1
          if last_l < j:
            last_l = j
        else:
          block[e[j]]+=1

        last_wl = wl
        j = j+1

      xpad = j

      for l in range(14):
        if False and seen[l] >= 0 and block[l] > 1:
          r = seen[l]
          pen = 0
          while r < last[l]:# and not pen:
            if e[seen[l]] != e[r]:
              pen+= 0.1
              if debug: print "36", labeldic[l], r
            r += 1
          if pen > 0:
            penalty[i] += 1.0

        #if e[seen[l]] == e[seen[l]+block[l]-1]:
        #  if block[l] > 1:
        #    reward[i] += 10
        #else:
        #  penalty[i] += 10
      #if block[aind] > 2:
      #  reward[i] += 5
      #if block[tind] > 5:
      #  reward[i] += 5
      #if block[jind] > 2:
      #  reward[i] += 5
      if seen[lpad] > 0:
        penalty[i] += 10


      #author must exist
      if seen[aind] < 0:
        penalty[i] += 1

      #date must exist
      if False and seen[dind] < 0:
        penalty[i] += 1

      if False and seen[lpad] >0 and seen[lpad] < xpad:
        penalty[i] += 1

        if debug: print "6"
      #if seen[bind] > 0 and seen[jind] > 0:
      #  penalty[i] += 1

      #One of the booktitle, journal, or tech
      if seen[bind] >= 0 :
        if seen[jind] >= 0 or  seen[techind] >=0:
          penalty[i] += 1
          if debug: print "7"

      if seen[jind] >= 0 :
        if seen[bind] >= 0 or  seen[techind] >=0:
          penalty[i] += 1
          if debug: print "7.1"

      if seen[techind] >= 0 :
        if seen[jind] >= 0 or  seen[bind] >=0:
          penalty[i] += 1
          if debug: print "7.2"


      #first tag is preferred to be author or editor
      if seen[aind] != 0:
        penalty[i] += 1
        if debug: print "8"


      #if seen[aind] > 0:
      #  if block[aind]

      #Journal

      #Booktitle
      #booktitle comes after title
      if seen[bind] >= 0 and seen[bind] < seen[tind]:
        penalty[i] += 1
        if debug: print "11"




      #location comes after journal or booktitle
      if seen[lind] >=0 and (seen[lind] < seen[jind] or seen[lind] < seen[bind]):
        penalty[i] += 1
        if debug: print "13"

      #if seen[lind] >=0  and seen[lind] > seen[tind]:
      #  reward[i] += 1

      #page
      #Page comes after booktitle or journal
      if seen[pind] >=0:
        if (seen[jind] >= 0 and seen[pind] < seen[jind]) or (seen[bind] >= 0 and seen[pind] < seen[bind]):
          penalty[i] += 1
          if debug: print "14"
      #volume
      #Volume comes after booktile or journal
      if seen[vind] >=0:
        if (seen[jind] >= 0 and seen[vind] < seen[jind]) or (seen[bind] >= 0 and seen[vind] < seen[bind]):
          penalty[i] += 1
          if debug: print "15"

        #if block[vind] > 5:
        #  penalty[i] += 1


      # note
      if False and seen[nind] >= 0:
        if seen[nind] < seen[aind]:
          penalty[i] += 1
          if debug: print "16"
        if seen[nind] < seen[tind]:
          penalty[i] += 1
          if debug: print "17"

      #publisher comes after journal or booktitle
      if seen[pubind] >= 0:
        if seen[pubind] < seen[jind] and seen[jind] >= 0:
          penalty[i] +=1
          if debug: print "18"
        if seen[pubind] < seen[bind] and seen[bind] >= 0:
          penalty[i] +=1
          if debug: print "19"

      #institution comes after author
      if False and seen[instind] >= 0:
        if seen[instind] < seen[aind]:
          penalty[i] +=1

        if seen[instind] < seen[tind]:
          penalty[i] += 1

      #tech
      if False and seen[techind] >= 0:
        if seen[techind] < seen[aind]:
          penalty[i] += 1

        if seen[techind] < seen[tind]:
          penalty[i] +=1


      #block size penalties:
      if seen[dind] >= 0 and (block[dind] > 5):# or block[dind] < 3):
        penalty[i] += 1.0

      if seen[tind] >=0 and (block[tind] > 20):# or block[tind] < 5):
        penalty[i] += 1.0

      if seen[lind] >= 0 and (block[lind] > 5):# or block[lind] < 3):
        penalty[i] += 1.0

      if seen[eind] >= 0 and (block[eind] > 6):# or block[eind] < 3) :
        penalty[i] += 1.0

      if seen[pubind] > 0 and (block[pubind] > 10):# or block[pubind] < 3):
        penalty[i] += 1.0

      if seen[bind] >= 0 and (block[bind] > 10):# or block[bind] < 3):
        penalty[i] += 1.0

      if seen[techind] >= 0 and (block[techind] > 5):# or block[techind] < 3):
        penalty[i] += 1.0

      if seen[jind] >= 0 and (block[jind] > 10):# or block[jind] < 3):
        penalty[i] += 1.0

      if seen[nind] >=0 and (block[nind] > 10):# or block[nind] < 3):
        penalty[i] += 1.0

      if seen[aind] >= 0 and (block[aind] > 15):# or block[aind] < 3):
        penalty[i] += 1.0

      if seen[instind] >= 0 and (block[instind] > 5):# or block[instind] < 3):
        penalty[i] += 1.0

      if seen[pind] >= 0 and (block[pind] > 7):# or block[pind] < 3):
        penalty[i] += 1.0

      if seen[vind] >= 0 and (block[vind] > 5):# or block[vind] < 3):
        penalty[i] += 1.0


      if block[lpad] > 0:
        penalty[i] += 1
        if debug: print "50.0"

      if False and seen[tind] >= 0:
        seenPeriodOnce = False

        for vx in block_x[tind]:
          if vx == '.':
            if seenPeriodOnce:
              penalty[i] += 0.1
            else:
              seenPeriodOnce = True



      if False and seen[vind] >= 0:
        found_number = False
        for vx in block_x[vind]:
          if vx.isdigit():
            found_number = True
            break

        if not found_number:
          if debug: print "51"
          penalty[i] += 0.1

      if False and seen[dind] >= 0:
        found_number = False
        for vx in block_x[dind]:
          if vx[:-1].isdigit():
            found_number = True
            break

        if not found_number:
          if debug: print "52"
          penalty[i] += 1.0


      if False and seen[pind] >= 0:
        found_number = False
        for vx in block_x[pind]:
          if vx.isdigit():
            found_number = True
            break

        if not found_number:
          if debug: print "53"
          penalty[i] += 0.1

      if debug and penalty[i] > 0:
        print " ".join(filter(lambda x: x != 'PAD', [worddic[k] for k in x]))
      if debug and penalty[i] > 0:
        print " ".join(filter(lambda x: x != 'PAD', [worddic[k] for k in x]))

  return -penalty





def evaluate_citation(xinput=None, yinput=None, yt=None):
  xd = xinput
  yd = yinput
  debug = False
  size = np.shape(xd)[0]

  if len(xd.shape) < 2:
    xd = np.expand_dims(xd, 0)
    yd = np.expand_dims(yd, 0)

  for i in range(size):

    try:
      pads = np.where(xd[i] == 0)
      non_pads = np.where(xd[i] > 0)
      first_pad = np.min(pads)
      last_pad = np.max(pads)
      first_nonpad = np.min(non_pads)
      last_nonpad = np.max(non_pads)
    except:
      continue

    if first_pad < first_nonpad and first_nonpad < last_nonpad and last_nonpad < last_pad:
      rol_val = -first_nonpad
    elif first_nonpad < last_nonpad and last_nonpad< first_pad and first_pad < last_pad:
      rol_val = 0
    elif first_nonpad < first_pad and first_pad < last_pad and last_pad < last_nonpad:
      rol_val = -(last_pad+1)
    elif first_pad < last_pad and last_pad < first_nonpad and first_pad < last_nonpad:
      rol_val = -first_nonpad
    else:
      rol_val = 0
    xd[i] = np.roll(xd[i], rol_val)
    yd[i] = np.roll(yd[i], rol_val)



  # if len(xd.shape) < 2:
  #   xd = np.expand_dims(xd, 0)
  #   yd = np.expand_dims(yd, 0)
  #
  # for i in range(xd.shape[0]):
  #   if xd[i][0] == 0:
  #     try:
  #       yd[i] = np.roll(yd[i], -np.where(xd != 0)[1][0])
  #       xd[i] = np.roll(xd[i], -np.where(xd != 0)[1][0])
  #     except:
  #       pass
  #   else:
  #     try:
  #       yd[i] = np.roll(yd[i], np.where(np.flip(xd[i], 0) == 0)[0][0])
  #       xd[i] = np.roll(xd[i], np.where(np.flip(xd[i], 0) == 0)[0][0])
  #
  #     except:
  #       pass

  reward = np.zeros(size)
  penalty = np.zeros(size)


  for i in range(size):
      seen = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1, 12: -1, 13: -1}
      last = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1, 12: -1, 13: -1}
      block = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1, 12: -1, 13: -1}
      block_x = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[]}
      e = yd[i,:]
      x = xd[i,:]
      last_l = 0
      xpad = 1000
      j = 0

      quote_open=False
      parenthesis_open = False
      dash = False
      last_wl = ""

      while (j < len(e) and x[j] != wpad):


        #if x[j] in wlocind and e[j] != locind:
        #  penalty[i] += 10
        #  if debug: print "41.1"
          #print worddic[x[j]], labeldic[e[j]]


        if x[j] in wmind:
          if e[j] != dind:
            penalty[i] += 10
            if debug: print "41.2"
          #else:
          #  print worddic[x[j]], labeldic[e[j]]

        word = worddic[x[j]]
        wl = word.lower()

        if e[j] == vind or e[j] == dind or e[j] == pind or e[j] == tind:
          block_x[e[j]].append(wl)

        if last_wl == '(' or wl == ')':
          if e[j] != e[j-1]:
            penalty[i] += 1
            if debug: print "41.3"


        if wl == '-' and last_wl == '-':
          if e[j] != pind:
            penalty[i] += 10
            if debug: print "41.4"

        if wl == '.':
          if j > 0 and e[j] != e[j-1]:
            penalty[i] += 1
            if debug: print "20.0"

        ## IM ADDING THEM HERE

        # publisher
        # if seen[pubind] >= 0:
        #     if wl == "press" or wl == "wiley" and e[j]!=pubind:
        #         penalty[i] +=2

        if word in title_dic and e[j] != tind:
          penalty[i] += 10

        elif word in a_list and e[j] != aind:
          penalty[i] += 10

        elif word in auth_dic and e[j] != aind:
          penalty[i] += 10

        elif word in uni_dic and e[j] != instind:
          penalty[i] += 10

        elif word in conf_list and e[j] != bind:
          penalty[i] += 10

        elif word in loc_dic and e[j] != locind:
          penalty[i] += 10




          #if (len(last_wl) == 1 and ord(last_wl) > ord('a') and ord(last_wl) < ord('z')) or (last_wl == 'pp' or last_wl == 'proc') :
            #print wl
          #  if j> 0 and j < output_num - 2 and e[j] != e[j+1]:
          #    penalty[i] += 1
          #    if debug: print "20.1"
              #print last_wl, labeldic[e[j-1]], labeldic[e[j]], labeldic[e[j+1]]
          #elif j < output_num - 2 and e[j] == e[j+1]:
          #  penalty[i] += 1
          #  if debug: print "20.2"
          #if j > 0 and j < output_num - 2:
          #  if e[j-1] == aind and e[j+1] == aind and e[j] != aind:
          #    penalty[i] += 1
          #    if debug: print "21"


        elif wl == '"' and e[j] != tind:
          penalty[i] += 10
          if debug: print "25"

        elif wl == ',':
          if e[j-1] == aind and e[j+1] == aind and e[j] != aind:
            penalty[i] += 1
            if debug: print "26"


        #Colon only comes in Title
        #elif wl == ':':
        #  if e[j] != tind:
        #    penalty[i] += 10
        #    if debug: print "27.1"
        #  if e[j] != e[j+1]:
        #    penalty[i] += 1
        #    if debug: print "27.2"
        #  if e[j-1] != e[j]:
        #    penalty[i] += 1
        #    if debug: print "27.3"
        #  if e[j-1] != e[j+1]:
        #    penalty[i] += 1
        #    if debug: print "27.4"



        elif wl.startswith('page') or wl == "pp":
          if e[j] != pind:
            penalty[i] += 10
            if debug: print "22"

        #add of, in,
        elif wl == "and" or wl =='-':
          if e[j-1] != e[j]:
            penalty[i] += 1
            if debug: print "23"
          if e[j] != e[j+1]:
            penalty[i] += 1
            if debug: print "24"





        #'The' is only comes in title, booktitle, or journal.
        #elif wl == "the":
        #  if e[j] != tind and e[j] != bind and e[j] != jind:
        #    penalty[i] += 1
        #    if debug: print "24.1"


        #elif wl == "for":
        #  if e[j] != tind:
        #    penalty[i] += 10
        #    if debug: print "24.2"

        elif wl.startswith("novel") or wl.startswith("method") \
            or wl.startswith("approach") or wl.startswith("technique") or wl.startswith("with")\
            or wl.startswith("propert") or wl.startswith("efficient"):
          if e[j] != tind:
            penalty[i] += 10
            if debug: print "24.3"

        elif (wl == "ijcai") and e[j] != bind:
          penalty[i] += 10
          if debug: print "24.4"

        elif wl =="ieee" or wl == "acm" or wl == "siam":
          if e[j] != bind and e[j] != jind:
            penalty[i] += 10
            if debug: print "27"

        elif wl.startswith("proceeding") or wl.startswith("conference") or wl == "proc":
          if e[j] != bind:
            penalty[i] += 10
            if debug: print "28"

        elif (wl.startswith("journal")):
          if e[j] != jind:
            penalty[i] += 10
            if debug: print "28.1"

        elif (wl.startswith("editor") or wl == "ed" ) and e[j] != eind:
          penalty[i] += 10
          if debug: print "29"

        elif (wl.startswith("techninal") or wl.startswith("preprint"))and e[j] != techind:
          penalty[i] += 10
          if debug: print "31"

        elif wl.startswith("date") and e[j] != dind:
          penalty[i] += 10
          if debug: print "32"

        elif (wl == "vol" or wl == "volume") and e[j] != vind:
          penalty[i] += 10
          if debug: print "32.1"


        elif (wl.startswith("university") or wl == "univ" or wl == "institute" ) and e[j] != instind:
          penalty[i] += 10
          if debug: print "33"

        #elif wl.startswith("note") and e[j] != nind:
          #penalty[i] += 10
        #  if debug: print "34"

        elif (len(wl) == 4 and year.search(wl) is not None) and e[j] != dind:
          penalty[i] += 10
          #print wl
          if debug: print wl
          if debug: print labeldic[e[j]]
          if debug: print "35"

        else:
          pass

          # try:
          #   if int(word) in range(50):
          #     if e[j] != vind:
          #       penalty[i] += 1
          # except:
          #   pass
          # try:
          #   if int(word) in range(1000, 1900):
          #     if e[j] != pind:
          #       penalty[i] += 1
          # except:
          #   pass


            #if j > 0 and j < output_num - 2 and e[j-1] != e[j+1]:
        #  if wl != '.' and wl != ',' and wl != '"' and wl != ')' and wl != '(':
        #    penalty[i] += 1
            #print wl, labeldic[e[j-1]], labeldic[e[j+1]]


        last[e[j]] = j
        if seen[e[j]] < 0:
          seen[e[j]] = j
          block[e[j]] = 1
          if last_l < j:
            last_l = j
        else:
          block[e[j]]+=1

        last_wl = wl
        j = j+1

      xpad = j

      for l in range(14):
        if seen[l] >= 0 and block[l] > 1:
          r = seen[l]
          pen = 0
          while r < last[l]:# and not pen:
            if e[seen[l]] != e[r]:
              pen+= 0.1
              if debug: print "36", labeldic[l], r
            r += 1
          if pen > 0:
            penalty[i] += 1.0

        #if e[seen[l]] == e[seen[l]+block[l]-1]:
        #  if block[l] > 1:
        #    reward[i] += 10
        #else:
        #  penalty[i] += 10
      #if block[aind] > 2:
      #  reward[i] += 5
      #if block[tind] > 5:
      #  reward[i] += 5
      #if block[jind] > 2:
      #  reward[i] += 5
      if seen[lpad] > 0:
        penalty[i] += 10


      #author must exist
      if False and seen[aind] < 0:
        penalty[i] += 1

      #date must exist
      if False and seen[dind] < 0:
        penalty[i] += 1

      if False and seen[lpad] >0 and seen[lpad] < xpad:
        penalty[i] += 1

        if debug: print "6"
      #if seen[bind] > 0 and seen[jind] > 0:
      #  penalty[i] += 1

      #One of the booktitle, journal, or tech
      if seen[bind] >= 0 :
        if seen[jind] >= 0 or  seen[techind] >=0:
          penalty[i] += 1
          if debug: print "7"

      if seen[jind] >= 0 :
        if seen[bind] >= 0 or  seen[techind] >=0:
          penalty[i] += 1
          if debug: print "7.1"

      if seen[techind] >= 0 :
        if seen[jind] >= 0 or  seen[bind] >=0:
          penalty[i] += 1
          if debug: print "7.2"


      #first tag is preferred to be author or editor
      if seen[aind] != 0:
        penalty[i] += 100
        if debug: print "8"


      #if seen[aind] > 0:
      #  if block[aind]

      #Journal

      #Booktitle
      #booktitle comes after title
      if seen[bind] >= 0 and seen[bind] < seen[tind]:
        penalty[i] += 1
        if debug: print "11"




      #location comes after journal or booktitle
      if seen[lind] >=0 and (seen[lind] < seen[jind] or seen[lind] < seen[bind]):
        penalty[i] += 1
        if debug: print "13"

      #if seen[lind] >=0  and seen[lind] > seen[tind]:
      #  reward[i] += 1

      #page
      #Page comes after booktitle or journal
      if seen[pind] >=0:
        if (seen[jind] >= 0 and seen[pind] < seen[jind]) or (seen[bind] >= 0 and seen[pind] < seen[bind]):
          penalty[i] += 1
          if debug: print "14"
      #volume
      #Volume comes after booktile or journal
      if seen[vind] >=0:
        if (seen[jind] >= 0 and seen[vind] < seen[jind]) or (seen[bind] >= 0 and seen[vind] < seen[bind]):
          penalty[i] += 1
          if debug: print "15"

        #if block[vind] > 5:
        #  penalty[i] += 1


      # note
      if False and seen[nind] >= 0:
        if seen[nind] < seen[aind]:
          penalty[i] += 1
          if debug: print "16"
        if seen[nind] < seen[tind]:
          penalty[i] += 1
          if debug: print "17"

      #publisher comes after journal or booktitle
      if seen[pubind] >= 0:
        if seen[pubind] < seen[jind] and seen[jind] >= 0:
          penalty[i] +=1
          if debug: print "18"
        if seen[pubind] < seen[bind] and seen[bind] >= 0:
          penalty[i] +=1
          if debug: print "19"



      #institution comes after author
      if False and seen[instind] >= 0:
        if seen[instind] < seen[aind]:
          penalty[i] +=1

        if seen[instind] < seen[tind]:
          penalty[i] += 1

      #tech
      if False and seen[techind] >= 0:
        if seen[techind] < seen[aind]:
          penalty[i] += 1

        if seen[techind] < seen[tind]:
          penalty[i] +=1


      #block size penalties:
      if seen[dind] >= 0 and (block[dind] > 5):# or block[dind] < 3):
        penalty[i] += 1.0

      if seen[tind] >=0 and (block[tind] > 20):# or block[tind] < 5):
        penalty[i] += 1.0

      if seen[lind] >= 0 and (block[lind] > 5):# or block[lind] < 3):
        penalty[i] += 1.0

      if seen[eind] >= 0 and (block[eind] > 6):# or block[eind] < 3) :
        penalty[i] += 1.0

      if seen[pubind] > 0 and (block[pubind] > 10):# or block[pubind] < 3):
        penalty[i] += 1.0

      if seen[bind] >= 0 and (block[bind] > 10):# or block[bind] < 3):
        penalty[i] += 1.0

      if seen[techind] >= 0 and (block[techind] > 5):# or block[techind] < 3):
        penalty[i] += 1.0

      if seen[jind] >= 0 and (block[jind] > 10):# or block[jind] < 3):
        penalty[i] += 1.0

      if seen[nind] >=0 and (block[nind] > 10):# or block[nind] < 3):
        penalty[i] += 1.0

      if seen[aind] >= 0 and (block[aind] > 15):# or block[aind] < 3):
        penalty[i] += 1.0

      if seen[instind] >= 0 and (block[instind] > 5):# or block[instind] < 3):
        penalty[i] += 1.0

      if seen[pind] >= 0 and (block[pind] > 7):# or block[pind] < 3):
        penalty[i] += 1.0

      if seen[vind] >= 0 and (block[vind] > 5):# or block[vind] < 3):
        penalty[i] += 1.0


      if block[lpad] > 0:
        penalty[i] += 1
        if debug: print "50.0"

      if False and seen[tind] >= 0:
        seenPeriodOnce = False

        for vx in block_x[tind]:
          if vx == '.':
            if seenPeriodOnce:
              penalty[i] += 0.1
            else:
              seenPeriodOnce = True



      if False and seen[vind] >= 0:
        found_number = False
        for vx in block_x[vind]:
          if vx.isdigit():
            found_number = True
            break

        if not found_number:
          if debug: print "51"
          penalty[i] += 0.1

      if False and seen[dind] >= 0:
        found_number = False
        for vx in block_x[dind]:
          if vx[:-1].isdigit():
            found_number = True
            break

        if not found_number:
          if debug: print "52"
          penalty[i] += 1.0


      if False and seen[pind] >= 0:
        found_number = False
        for vx in block_x[pind]:
          if vx.isdigit():
            found_number = True
            break

        if not found_number:
          if debug: print "53"
          penalty[i] += 0.1

      if debug and penalty[i] > 0:
        print " ".join(filter(lambda x: x != 'PAD', [worddic[k] for k in x]))

  return -penalty



def perf(ytr_pred, yval_pred, yts_pred, ydata, yval, ytest):
  global best_val
  global test_val

  hm_ts, ex_ts = token_level_loss(yts_pred, ytest)
  hm_tr, ex_tr = token_level_loss(ytr_pred, ydata)
  hm_val, ex_val = token_level_loss(yval_pred, yval)
  if ex_val > best_val:
    best_val = ex_val
    test_val = ex_ts
  print (" ----------------------  Train: %0.3f Val: %0.3f Test: %0.3f ------ Best Val: %0.3f Test: %0.3f ------------" % (
      ex_tr, ex_val, ex_ts, best_val, test_val))


def check(xd, yd):
  r = evaluate_citation(xinput=xd, yinput=yd)
  return np.average(r)

def train(spen, num_steps):
  global ln
  global bs
  global xdata, xtest, xval, ydata, ytest, yval, x_unlab
  global embedding

  ntrain = np.shape(xdata)[0]
  spen.init()
  spen.init_embedding(embedding)

  labeled_num = min((ln, ntrain))
  indices = np.arange(labeled_num)

  xorig = xdata[indices]
  yorig = ydata[indices]

  rotatedx = xorig[:]
  rotatedy = yorig[:]
  for i in range(-50, 50, 5):
    xshift = np.roll(xorig, i, 1)
    yshift = np.roll(yorig, i, 1)
    rotatedx = np.vstack((rotatedx, xshift))
    rotatedy = np.vstack((rotatedy, yshift))




  #xlabeled = rotatedx
  #ylabeled = rotatedy

  xwhole = np.vstack((xdata, x_unlab))
  #xwhole = rotatedx
  #xwhole = xdata
  total_num = np.shape(xwhole)[0]

  pen = evaluate_citation(xinput=xdata, yinput=ydata)
  rotated = evaluate_citation(xinput=rotatedx, yinput=rotatedy)
  print ("ydata pen:", np.average(pen))
  print ("ydata rotated pen:", np.average(rotated))

  for i in range(1, num_steps):
    bs = min((bs, labeled_num))
    #bs = bs + 10
    perm = np.random.permutation(total_num)

    for b in range(total_num / bs):

      indices = perm[b * bs:(b + 1) * bs]

      xbatch = xwhole[indices][:]



      spen.set_train_iter(i)
      o = spen.train_batch(xbatch=xbatch, verbose=1)


      #print (i, b, o, bs)


    yts_out = spen.map_predict(xtest)
    yval_out = spen.map_predict(xval)
    ytr_out = spen.map_predict(xdata)
    perf(ytr_out, yval_out, yts_out, ydata, yval, ytest)
    gt = check(xdata, ydata)
    pe = check(xdata, ytr_out)
    pt = check(xtest, ytest)
    te = check(xtest, yts_out)
    print("Score: %d %0.3f %.3f %0.3f %0.3f" % (i, gt, pe, pt, te))

    r = np.random.randint(0, 99, 2)
    x_db = np.squeeze(xdata[r,:])
    y_db = spen.map_predict(x_db)
    for j in range(0):
      a = [worddic[t] for t in x_db[j]]
      b = [labeldic[t] for t in y_db[j]]
      c = filter(lambda (x, y): x != 'PAD', zip(a, b))
      print(" ".join(filter(lambda x: x != 'PAD', a)))
      print(c)


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

config.inf_penalty = ip



s = sp.SPEN(config)
e = energy.EnergyModel(config)

#s.eval = lambda xd, yd, yt : token_level_loss_ar(yd, yt)
s.get_energy = e.get_energy_mlp_emb
s.train_batch = s.train_unsupervised_batch
s.evaluate = evaluate_citation

s.createOptimizer()
s.construct_embedding(embedding_size,vocabulary_size)
s.construct(training_type=sp.TrainingType.Rank_Based)
s.print_vars()

start = time.time()
train(s, 10000)


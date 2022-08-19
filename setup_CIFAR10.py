from utils import dataload
from utils import utils
from utils import config

import numpy as np
import os
import argparse


import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

c = config.Configuration()
args = c.getArgs()

print(f"hello MIAT - we are creating CIFAR10 data on disk as np arrays")

X = []
Y = []

for i in range(1,6):
    name = "data_batch_" + str(i)
    data_dict = dataload.unpickle(os.path.join("./data/cifar-10-batches-py", name))
    X_tmp = data_dict[b'data']
    Y_tmp = data_dict[b'labels']
    X.append(X_tmp) 
    Y.append(Y_tmp)
#reshape makes it [sample][channel][row][col]
#transpose makes it [sample][row][col][channel]
X = np.asarray(X, dtype='uint8').reshape(50000,3, 32, 32).transpose(0,2,3,1)
Y = np.asarray(Y, dtype='uint8').reshape(50000)
print("X shape:",np.shape(X))
print("Y shape:",np.shape(Y))
label_dict = dataload.unpickle_string("./data/cifar-10-batches-py/batches.meta")

data_dict_test = dataload.unpickle("./data/cifar-10-batches-py/test_batch")
X_test = data_dict[b'data']
Y_test = data_dict[b'labels']

X_test = np.asarray(X_test, dtype='uint8').reshape(10000,3,32,32).transpose(0,2,3,1)
Y_test = np.asarray(Y_test, dtype='uint8').reshape(10000)

print("X_test shape:",np.shape(X_test))
print("Y_test shape:",np.shape(Y_test))

#index into classes for the correct label
classes = label_dict['label_names']  #['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
utils.displayRandomImgGrid(X, Y, classes, rows=5, cols=5, Y_hat=None)

#https://numpy.org/doc/stable/reference/generated/numpy.save.html

print("Saving numpy arrays to file ...")
with open(args.nat_img_train, 'wb') as f:
    np.save(f, X)
with open(args.nat_label_train, 'wb') as f:
    np.save(f, Y)
with open(args.nat_img_test, 'wb') as f:
    np.save(f,X_test)
with open(args.nat_label_test, 'wb') as f:
    np.save(f,Y_test)
print("Save complete!")


print(f"Hello again MIAT, EOF")

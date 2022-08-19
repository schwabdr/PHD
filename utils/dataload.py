'''
File to support loading / saving of data to disk
'''

import torch
from torch.utils.data import Dataset, DataLoader
import re
import pickle
from PIL import Image
import os
import numpy as np

#not sure why the original code doesn't have the code to load the data - but I'm going to try and add it
#http://www.cs.toronto.edu/~kriz/cifar.html - all about CIFAR10

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def unpickle_string(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='utf-8')
    return dict


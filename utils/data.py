import sys
'''
Helper functions to load datasets
'''

import numpy as np
import torch.utils.data as Data
from PIL import Image

# import tools
import torch
import utils.utils

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import random

# subclass of the abstract torch.utils.data.Dataset class
# map style?
class data_dataset(Data.Dataset):
    def __init__(self, img_path, clean_label_path, transform=None):
        self.transform = transform
        self.train_data = np.load(img_path)
        self.train_clean_labels = np.load(clean_label_path).astype(np.float32)
        self.train_clean_labels = torch.from_numpy(self.train_clean_labels).long()
    #this function below does return the correct labels - I'm not sure if the image is correct.
    #still not sure I've put the data in the correct format on disk to be read in.
    def __getitem__(self, index):
        #print("__getitem__() inside data.py:")
        img, clean_label = self.train_data[index], self.train_clean_labels[index]
        #print("img (before fromarray): ", img)
        img = Image.fromarray(img)
        #print("img (after fromarray): ", img)
        if self.transform is not None:
            img = self.transform(img)
        #print("img (after transform): ", img)
        #util.print_tensor_details("img", img)
        #print(f"img: {img}")
        #util.print_tensor_details("clean_label", clean_label)
        #print(f"clean_label: {clean_label}")
        return img, clean_label

    def __len__(self):
        return len(self.train_data)


class data_adv_dataset(Data.Dataset):
    def __init__(self, img_path, adv_img_path, clean_label_path, transform=None, augment=False):
        self.transform = transform
        self.train_data = np.load(img_path)
        self.adv_train_data = np.load(adv_img_path)
        self.train_clean_labels = np.load(clean_label_path).astype(np.float32)
        self.train_clean_labels = torch.from_numpy(self.train_clean_labels).long()
        self.augment = augment
    #this function below does return the correct labels - I'm not sure if the image is correct.
    #still not sure I've put the data in the correct format on disk to be read in.
    #see here for how to apply same transform to both images
    #https://discuss.pytorch.org/t/how-to-apply-same-transform-on-a-pair-of-picture/14914?u=ssgosh
    def __getitem__(self, index):
        img, adv_img, clean_label = self.train_data[index], self.adv_train_data[index], self.train_clean_labels[index]
        img = Image.fromarray(img)
        adv_img = Image.fromarray(adv_img)
        
        if self.augment:
            # Resize
            resize = transforms.Resize(size=(36,36))
            img = resize(img)
            adv_img = resize(adv_img)
            
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(32, 32))
            img = TF.crop(img, i, j, h, w)
            adv_img = TF.crop(adv_img, i, j, h, w)

            #Horizontal Flip
            if random.random() > 0.5:
                img = TF.hflip(img)
                adv_img = TF.hflip(adv_img)

        if self.transform is not None: #performs normalization
            img = self.transform(img)
            adv_img = self.transform(adv_img)
        
        
        return img, adv_img, clean_label

    def __len__(self):
        return len(self.train_data)

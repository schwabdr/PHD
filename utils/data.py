import sys
'''
Helper functions to load datasets
'''

import numpy as np
import torch.utils.data as Data
from PIL import Image

# import tools
import torch

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

'''
This class is for getting pairs of images that are a clean image and an adversarial example precomputed for that image.
'''
class data_adv_dataset(Data.Dataset):
    def __init__(self, img_path, adv_img_path, clean_label_path, transform=None, augment=False):
        self.transform = transform
        self.train_data = np.load(img_path)
        self.adv_train_data = np.load(adv_img_path)
        self.train_clean_labels = np.load(clean_label_path).astype(np.float32)
        self.train_clean_labels = torch.from_numpy(self.train_clean_labels).long()
        self.augment = augment
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


'''
The purpose of this class is to return two images from two different classes.
Also returns the label for each image.

These are all clean images / labels.
'''
class data_micraft_dataset(Data.Dataset):
    def __init__(self, img_path, clean_label_path, transform=None, augment=False):
        self.transform = transform
        self.train_data = np.load(img_path)
        self.train_data = np.load(img_path)
        self.train_labels = np.load(clean_label_path).astype(np.float32)
        self.train_labels = torch.from_numpy(self.train_labels).long()
        
        self.augment = augment
        self.length = len(self.train_data)
    #see here for how to apply same transform to both images
    #https://discuss.pytorch.org/t/how-to-apply-same-transform-on-a-pair-of-picture/14914?u=ssgosh
    def __getitem__(self, index):
        img_clean, label_clean = self.train_data[index], self.train_labels[index]
        i_target = random.randint(0,self.length-1)
        label_target = self.train_labels[i_target]

        while(label_clean == label_target): #while clean class and target class are the same
            i_target = random.randint(0,self.length-1)
            label_target = self.train_labels[i_target]
        
        img_target = self.train_data[i_target]
        
        img_clean = Image.fromarray(img_clean)
        img_target = Image.fromarray(img_target)
        
        if self.augment:
            # Resize
            resize = transforms.Resize(size=(36,36))
            img_clean = resize(img_clean)
            img_target = resize(img_target)
            
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(img_clean, output_size=(32, 32)) #only do this once since all images are same size - may not be true for another dataset
            img_clean = TF.crop(img_clean, i, j, h, w)
            img_target = TF.crop(img_target, i, j, h, w)

            #Horizontal Flip
            if random.random() > 0.5:
                img_clean = TF.hflip(img_clean)
                img_target = TF.hflip(img_target)

        if self.transform is not None: #performs normalization
            img_clean = self.transform(img_clean)
            img_target = self.transform(img_target)
       
        return img_clean, img_target, label_clean, label_target

    def __len__(self):
        return self.length

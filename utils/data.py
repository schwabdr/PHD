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
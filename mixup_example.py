'''
Purpose of this file is to play around with ImageNet dataset and generate just one adversarial example at a time, visualize the adv ex and original image side by side.
'''
import projected_gradient_descent as pgd

from torchvision.io import read_image
from torchvision.models import resnet152, ResNet152_Weights

import torch

import numpy as np

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

#https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
def crop_center(img, cropx, cropy):
    y, x, *_ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    return img[starty:starty + cropy, startx:startx + cropx, ...]

'''
#same source as above - you have to import operator
I want to look at both of these and understand them.
def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]
'''

img1 = read_image("data/tabby03.png")
img2 = read_image("data/dog-02.jpeg")
img1 = img1.numpy().transpose(1,2,0) #[H][W][Channel]
img2 = img2.numpy().transpose(1,2,0)

img1 = crop_center(img1, 450, 300)

print(f"img1: {img1.shape}")
print(f"img2: {img2.shape}")

img1 = img1/255
img2 = img2/255

img3 = .6 * img1 + .4 * img2
#print(img3)

'''
img_adv_show = img_adv.detach().cpu().numpy().transpose(0,2,3,1)
img_clean_show = batch.detach().cpu().numpy().transpose(0,2,3,1)

#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])

stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # mean 
#x = np.clip(((x * stats[1]) + stats[0]),0,1.)
img_clean_show[0] = np.clip(((img_clean_show[0] * stats[1]) + stats[0]),0,1.)
img_adv_show[0] = np.clip(((img_adv_show[0] * stats[1]) + stats[0]),0,1.)

'''

fig,axes1 = plt.subplots(1,3,figsize=(10,10))
axes1[0].set_axis_off()
axes1[1].set_axis_off()
axes1[2].set_axis_off()
axes1[0].imshow(img1, interpolation='nearest')
axes1[1].imshow(img2, interpolation='nearest')
axes1[2].imshow(img3, interpolation='nearest')



plt.show()

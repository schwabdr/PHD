import os
import numpy as np
from PIL import Image

#next is only needed to visualize samples
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt



#def crop_center(self, img, cropx, cropy):
#       _, y, x = img.shape
#       startx = x // 2 - (cropx // 2)
#       starty = y // 2 - (cropy // 2)
#       return img[:, starty:starty + cropy, startx:startx + cropx]

def get_img(img, adv=False, show=False):
    if not adv:
        start_row = 501
    else: #adversarial example
        start_row = 600
    start_col = 522
    dx, dy = 100,94
    new_img = img[start_col:start_col+dx,start_row:start_row+dy,0:4]
    if show:
        print(f"shape: {np.shape(new_img)}")
        plt.imshow(new_img)
        plt.show()
    return new_img


full_names = []
folders = ["./test1-pgd-ce-5.1/", "./test1-pgd-ce-tar-5.2/", "./test2-cos-mc-5.3/", "./test2-euc-mc-5.4/"]
model_names = ['resnet-new-100', 'resnet-new-100-MIAT-from-scratch', 'resnet-new-100-MIAT-0.1-from-scratch', 'resnet-new-100-MIAT-0.25-from-scratch']
eps_lst = [.025, .05, .075, .1, .125, .15, .175, .2, .25, .3, .4, .5, .75, 1.] #14 values
suffixes = ["PGD-CE", "PGD-CE-T", "MI-COS", "MI-EUC"]

for (folder, suffix) in zip(folders, suffixes):
    for name in model_names:
        for eps in eps_lst:
            full_names.append(os.path.join(folder, str(eps) + "-" + name + "-" + suffix + ".PNG"))

#for full_name in full_names:
#    print(full_name)


#https://stackoverflow.com/questions/31386096/importing-png-files-into-numpy
im_frame = Image.open('./test1-pgd-ce-5.1/0.05-resnet-new-100-PGD-CE.PNG')
np_frame = np.array(im_frame)
im_frame.close()

#print(f"shape: {np.shape(np_frame)}") # [2500,2500,4]
#print(f"{np_frame}")
#plt.imshow(np_frame)
#plt.show()
# im_adv = get_img(np_frame, adv=True)
# im2 = get_img(np_frame, adv=False)

# im3 = np.concatenate((im2, im_adv), axis=0) #axis=0 stacks vertically, axis=1 stacks horizontally
# plt.imshow(im3)
# plt.show()

clean_img = get_img(np_frame, adv=False)
clean_col = clean_img

for _ in range(13):
    clean_col = np.concatenate((clean_col, clean_img), axis=0)

final_img = clean_col #initialize to clean column
#idea - construct each column, then concat horizontally - axis=1
first = True
for i, f_name in enumerate(full_names):
    im_frame = Image.open(f_name)
    np_frame = np.array(im_frame)
    im_frame.close()
    if first:
        new_col = get_img(np_frame, adv=True)
        first = False
    else:
        new_col = np.concatenate((new_col, get_img(np_frame, adv=True)), axis=0)
    #have a new column of images, add to final output
    if (i+1)%14 == 0:
        final_img = np.concatenate((final_img, new_col), axis=1)
        first = True
plt.imshow(final_img)
plt.show()




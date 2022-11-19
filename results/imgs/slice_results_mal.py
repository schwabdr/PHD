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
        start_row = 53
    else: #adversarial example
        start_row = 264
    start_col = 143
    dx, dy = 205,200
    new_img = img[start_col:start_col+dx,start_row:start_row+dy,0:4]
    if show:
        print(f"shape: {np.shape(new_img)}")
        plt.imshow(new_img)
        plt.show()
    return new_img


full_names = []
folders = ["./test1-mal-pgd-ce-5.1/", "./test1-mal-pgd-ce-tar-5.2/", "./test2-mal-cos-mc-5.3/", "./test2-mal-euc-mc-5.4/"]
#model_names = ['resnet-mal-std-100', 'resnet-mal-std-aug-100', 'resnet-mal-MIAT.25', 'resnet-mal-MIAT-AT.25.40']
model_names = ['resnet-mal-std-100', 'resnet-mal-std-aug-100'] 
#model_names = ['resnet-mal-MIAT.25', 'resnet-mal-MIAT-AT.25.40']
#eps_lst = [.025, .05, .075, .1, .125, .15, .175, .2, .25, .3, .4, .5, .75, 1.] #14 values
eps_lst = [.025, .1, .15, .2, .3, .4, .5, 1.] #
num_eps = len(eps_lst)
suffixes = ["PGD-CE", "PGD-CE-T", "MI-COS", "MI-EUC"]

for (folder, suffix) in zip(folders, suffixes):
    for name in model_names:
        for eps in eps_lst:
            full_names.append(os.path.join(folder, str(eps) + "-" + name + "-" + suffix + ".PNG"))

#for full_name in full_names:
#    print(full_name)

#https://stackoverflow.com/questions/31386096/importing-png-files-into-numpy
im_frame = Image.open('./test1-mal-pgd-ce-5.1/0.1-resnet-mal-std-100-PGD-CE.PNG')
np_frame = np.array(im_frame)
im_frame.close()

print(f"shape: {np.shape(np_frame)}") # [500,500,4]
#print(f"{np_frame}")
#plt.imshow(np_frame)
#plt.show()
#im_adv = get_img(np_frame, adv=True)
#im2 = get_img(np_frame, adv=False)

#im3 = np.concatenate((im2, im_adv), axis=0) #axis=0 stacks vertically, axis=1 stacks horizontally
#plt.imshow(im3)
#plt.show()

clean_img = get_img(np_frame, adv=False)
clean_col = clean_img

for _ in range(num_eps-1):
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
    if (i+1)%num_eps == 0:
        final_img = np.concatenate((final_img, new_col), axis=1)
        first = True
#https://stackoverflow.com/questions/24185083/change-resolution-of-imshow-in-ipython
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.imshow(final_img)
#plt.imshow(final_img)
#plt.show()
#https://stackoverflow.com/questions/9295026/how-to-remove-axis-legends-and-white-padding
plt.axis('off')
fig.savefig('./mal-grid/std-mal-grid-raw.PNG', dpi=1000) #, bbox_inches='tight')




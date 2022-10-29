import tkinter

tkinter._test()

import os
import numpy as np
from PIL import Image

#next is only needed to visualize samples
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

im_frame = Image.open('./results/imgs/test1-pgd-ce-5.1/0.05-resnet-new-100-PGD-CE.PNG')
np_frame = np.array(im_frame)
im_frame.close()

plt.imshow(np_frame)
plt.show()
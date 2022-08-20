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


img = read_image("data/tabby03.png")
# Step 1: Initialize model with the best available weights
#weights = ResNet50_Weights.DEFAULT
#model = resnet50(weights=weights)
weights = ResNet152_Weights.DEFAULT
model = resnet152(weights=weights)

model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)
#print(batch)
# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

saved_id = class_id

eps = .03
eps_iter = .007
nb_iter = 100
print(f"Using PGD with eps: {eps}, eps_iter: {eps_iter}, nb_iter: {nb_iter}")
#281: tabby cat, 933: cheese burger
img_adv = pgd.projected_gradient_descent(model, batch, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, y=torch.tensor([933]), targeted=True)
#print("img_adv: ",img_adv)

prediction = model(img_adv).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

#find and print the category / confidence for the original prediction now of the adversarial example:
score = prediction[saved_id].item()
print("confidence on original class label for the adversarial example:")
category_name = weights.meta["categories"][saved_id]
print(f"{category_name}: {100 * score:.1f}%")

#print("weights.meta[categories]: ", weights.meta["categories"][933]) #933 should be cheeseburger

img_adv_show = img_adv.detach().cpu().numpy().transpose(0,2,3,1)
img_clean_show = batch.detach().cpu().numpy().transpose(0,2,3,1)

#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])

stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # mean 
#x = np.clip(((x * stats[1]) + stats[0]),0,1.)
img_clean_show[0] = np.clip(((img_clean_show[0] * stats[1]) + stats[0]),0,1.)
img_adv_show[0] = np.clip(((img_adv_show[0] * stats[1]) + stats[0]),0,1.)


fig,axes1 = plt.subplots(1,2,figsize=(10,10))
axes1[0].set_axis_off()
axes1[1].set_axis_off()
axes1[0].imshow(img_clean_show[0], interpolation='nearest')
axes1[1].imshow(img_adv_show[0], interpolation='nearest')

plt.show()
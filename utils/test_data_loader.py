from data import data_mal_sample_dataset
import config
import torch
from torchvision import transforms

args = config.Configuration().getArgs()

stats = ((0.4454, 0.4454, 0.4454), (0.3122, 0.3122, 0.3122)) #mean and stdev

trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=False)
    ])

#testset=data_mal_sample_dataset(img_path=args.nat_img_test_mal, clean_label_path=args.nat_label_test_mal)
testset=data_mal_sample_dataset(img_path="../data/test_images_mal.npy", clean_label_path="../data/test_labels_mal.npy", transform=trans)
#note batchsize = 1 below because we get back a list of 25 images at once - we only need 1 batch.
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, drop_last=False, shuffle=False, num_workers=0, pin_memory=True)

for x_natural, y_true in test_loader:
    print(f"x_natural: {x_natural}")
    print(f"y_true: {y_true}")

    print(f"x_natural {len(x_natural)}")
    print(f"y_true {len(y_true)}")
    print(f"x_natural[5].size(): {x_natural[5].size()}")
    break

    

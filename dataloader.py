import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
# import timm
import torch.nn.functional as F
import os
from pickle import *
from sklearn.metrics import *


data_transforms = {
	"train": transforms.Compose([transforms.ToTensor(),
									transforms.Resize(224, antialias =True)]),
	"new_val": transforms.Compose([transforms.ToTensor(), transforms.Resize(224, antialias = True)]),
	# "test":transforms.Compose([transforms.ToTensor()])

}

data_dir = "/ssd_scratch/cvit/shaon/imageNet-tiny/tiny-imagenet-200"

image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
					for x in ["train", "new_val"]}


batch_size = 200
train_dataloaders = DataLoader(image_datasets["train"], batch_size = batch_size, shuffle = "True", num_workers = 8)
val_dataloaders = DataLoader(image_datasets["new_val"], batch_size = batch_size, shuffle = "True", num_workers = 8)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data.class_to_idx
# classes_names_val = image_datasets["new_val"].class_to_idx
# classes_names_train = image_datasets["train"].class_to_idx
# print("validation : ",classes_names_val)
# print("train:", classes_names_train)

for batch_index, (data, target) in enumerate(val_dataloaders):
	data = data.to(device = device)
	target = target.to(device = device)
	print(data.shape)
	print(target)
	break


	









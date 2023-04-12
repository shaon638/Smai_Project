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

import torch.optim as optim
import time
import numpy as np
from vit_pytorch import ViT


data_transforms = {
	"train": transforms.Compose([transforms.ToTensor(), transforms.Resize(224, antialias = True)]),
	"new_val": transforms.Compose([transforms.ToTensor(), transforms.Resize(224 , antialias = True)]),
	
}

data_dir = "/ssd_scratch/cvit/shaon/imageNet-tiny/tiny-imagenet-200"

image_datatsets = {x : datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
					for x in ["train", "new_val"]}
batch_size = 32

train_dataloaders = DataLoader(image_datatsets["train"], batch_size = batch_size, shuffle = "True", num_workers = 8)
val_dataloaders = DataLoader(image_datatsets["new_val"], batch_size = batch_size, shuffle = "True", num_workers = 8)

def getOptimizer(model, lr, mode, momentum = 0.09, weight_decay = 1e-4):
  if mode == "SGD":
    optimizer = optim.SGD(model.parameters(), lr = lr)
  elif mode == "SGD_M":
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
  elif mode == "SGD_L2":
    optimizer = optim.SGD(model.parameters(), lr = lr , weight_decay = weight_decay)
  elif mode == "RMS":
    optimizer =optim.RMSprop(model.parameters(), lr=lr)
  elif mode == "Adam":
    optimizer = optim.Adam(model.parameters(), lr = lr)
  return optimizer, mode

def eval_model(model, val_loader, criterion,device = "cuda"):
	out_loss_val = 0
	model.eval()

	for batch_index, (data, target) in enumerate(val_loader):
		data = data.to(device = device)
		target = target.to(device = device)

		with torch.no_grad():
			score = model(data)
		loss = criterion(score, target)
		out_loss_val += loss.item()

		if batch_index % 100 == 0:
			print(f"validationBatchLoss:{batch_index}\t loss :{out_loss_val/(batch_index+1)}")
	return out_loss_val


def train(model,num_epochs,train_loader,val_loader,optimizer,criterion, model_name = "Small_VIT_ImageNet_tiny", model_path = "/ssd_scratch/cvit/shaon/teacher_model",loss_path= "teacher_model_loss", device = "cuda"):
	train_loss = {}
	val_loss = {}
	dur = []

	for epoch in range(num_epochs):
		t0 = time.time()
		out_loss = 0
		model.train()

		for batch_index, (data,target) in enumerate(train_loader):
			optimizer.zero_grad()
			data = data.to(device = device)
			target = target.to(device = device)
			scores = model(data)
			loss = criterion(scores, target)
			loss.backward()
			optimizer.step()
			out_loss += loss.item()
			if batch_index % 100 == 0:
  				print(f"TrainBatchLoss: {batch_index}\t loss :{out_loss/(batch_index+1)}",flush = True)
		train_loss[epoch+1] = (out_loss/len(train_loader))
		torch.save(model.state_dict(),model_path+"/"+model_name +"_" +str(num_epochs))
		out_loss_validation = eval_model(model, val_loader, criterion)
		val_loss[epoch+1] = (out_loss_validation/len(val_loader))

		dur.append(time.time() - t0)
		curr_lr = optimizer.param_groups[0]['lr']
		print(f'Epoch {epoch+1} \t Training Loss: {out_loss / len(train_loader)} \t Validation Loss : {out_loss_validation/len(val_loader)}\t LR:{curr_lr} \t Time(s):{np.mean(dur)}', flush = True)

		torch.save(model.state_dict(),model_path+"/"+model_name +"_" +str(num_epochs))


		with open (loss_path+ "/train_loss"+"_"+model_name+"_"+str(num_epochs)+".pkl","wb") as file:
			dump(train_loss, file)
		with open(loss_path + "/val_loss"+"_"+model_name+"_"+str(num_epochs)+".pkl","wb")as file:
			dump(val_loss, file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



vision_transformer = ViT(
    image_size = 224,
    patch_size = 32,
    num_classes = 200,
    dim = 1024,
    depth = 3,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer, _ = getOptimizer(vision_transformer, 0.001, "Adam")
num_epochs = 1000
train(vision_transformer, num_epochs, train_dataloaders, val_dataloaders, optimizer, criterion)
# print(v)
# total_params = sum(
# 	param.numel() for param in vision_transformer.parameters()
# )
# print(total_params)

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

transforms = transforms.Compose([transforms.ToTensor(), 
									transforms.Resize(224,antialias = True),
									transforms.RandomHorizontalFlip(p=0.9), 
									transforms.RandomVerticalFlip(p=0.9), 
									transforms.RandomRotation(degrees=10),
									transforms.Normalize((0.2675, 0.2565, 0.2761),(0.5071, 0.4867, 0.4408))])




trainset =datasets.CIFAR100(root = "/ssd_scratch/cvit/shaon", train = True, transform =transforms ,download = True)

trainset, valset = torch.utils.data.random_split(trainset, [48000, len(trainset)-48000])



# testset = datasets.CIFAR100(root = "/ssd_scratch/cvit/shaon", train = False,transform = transforms.ToTensor() ,download = True)

batch_size = 16
train_loader = DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 8)
val_loader = DataLoader(valset, batch_size = batch_size, shuffle = True, num_workers = 8)
# test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
# val_loader = DataLoader(valset, batch_size = batch_size, shuffle = True, num_workers = 8)


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

		if batch_index % 10 == 0:
			print(f"validationBatchLoss:{batch_index}\t loss :{out_loss_val/(batch_index+1)}", flush = True)

	return out_loss_val


def inference_model(model, val_loader, criterion,device= "cuda"):
	predicted = []
	ground_truth = []

	model.eval()
	with torch.no_grad():
		for batch_index, (data, gt) in enumerate(val_loader):
			data = data.to(device = device)
			gt = gt.to(device = device)

			scores= model(data)
			_, predictiones = scores.max(1)
			pred = predictiones.cpu().detach().numpy()
			gt = gt.cpu().detach().numpy()
			predicted += list(pred)
			ground_truth += list(gt)

		total_accuracy = accuracy_score(ground_truth, predicted)*100
		return total_accuracy

def train(model,num_epochs,train_loader,val_loader,optimizer,criterion, model_name = "VIT_small_cifar100", model_path = "/ssd_scratch/cvit/shaon/teacher_model",loss_path= "loss", device = "cuda"):
	train_loss = {}
	val_loss =  {}
	dur = []
	best_vaild_loss = float('inf')
	lines= []

	for epoch in range(num_epochs):
		t0 = time.time()

		out_loss = 0
		model.train()
		for batch_index, (data, target) in enumerate(train_loader):

  			optimizer.zero_grad()
  			data = data.to(device = device)
  			target = target.to(device = device)
  			scores = model(data)

  
  			loss = criterion(scores, target)


  			loss.backward()

  			#gradient descent
  			optimizer.step()


  			out_loss += loss.item()
  			if batch_index % 100 == 0:
  				print(f"TrainBatchLoss: {batch_index}\t loss :{out_loss/(batch_index+1)}",flush = True)

  	




		train_loss[epoch+1] = (out_loss/len(train_loader))


		out_loss_val  = eval_model(model, val_loader,criterion,device)
		val_acc = inference_model(model, val_loader, criterion, device)
 


		val_loss[epoch+1] = (out_loss_val/len(val_loader))
		# current_val_loss = (out_loss_val/len(val_loader))

		dur.append(time.time() - t0)
		curr_lr = optimizer.param_groups[0]['lr']

		print(f'Epoch {epoch+1} \t Training Loss: {out_loss / len(train_loader)} \t Valid_Loss: {out_loss_val/len(val_loader)} \t val_acc@1: {val_acc} \t LR:{curr_lr} \t Time(s):{np.mean(dur)}', flush = True)

		# single_line = "Epoch:" + str(epoch+1) + "train_loss:" + str(out_loss/len(train_loader)) + "val_loss:" + str(out_loss_val/len(val_loader)) + "val_acc@1" + str(val_acc) + "Time in second" + str(np.mean(dur))
		# lines.append(single_line)




		# if current_val_loss < best_vaild_loss:
		# 	best_vaild_loss = current_val_loss
		# 	print(f"\nSaving the best model for epoch:{epoch +1}\n", flush = True)

		torch.save(model.state_dict(),model_path+"/"+model_name +"_" +str(epoch+1))


		with open (loss_path+ "/train_loss"+"_"+model_name+"_"+str(epoch +1)+".pkl","wb") as file:
			dump(train_loss, file)
		with open(loss_path + "/val_loss"+"_"+model_name+"_"+str(epoch+1)+".pkl","wb")as file:
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
train(vision_transformer, num_epochs, train_loader, val_loader, optimizer, criterion)
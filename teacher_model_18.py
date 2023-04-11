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
# import pycls.core.builders as builders



data_transforms = {
	"train": transforms.Compose([transforms.ToTensor(), transforms.Resize(224, antialias = True)]),
	"val": transforms.Compose([transforms.ToTensor(), transforms.Resize(224 , antialias = True)]),
	"test":transforms.Compose([transforms.ToTensor(), transforms.Resize(224, antialias = True)])
}

data_dir = "/ssd_scratch/cvit/shaon/imageNet-tiny/tiny-imagenet-200"

image_datatsets = {x : datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
					for x in ["train", "val", "test"]}

# print(image_datatsets)
batch_size = 200

train_dataloaders = DataLoader(image_datatsets["train"], batch_size = batch_size, shuffle = "True", num_workers = 8)
val_dataloaders = DataLoader(image_datatsets["val"], batch_size = batch_size, shuffle = "False", num_workers = 8)

# test_dataloaders = DataLoader(image_datatsets["test"], batch_size = batch_size, shuffle = "False", num_workers = 2)

dataset_sizes = {x : len(image_datatsets[x]) for x in ["train", "val", "test"]}
# print(dataset_sizes)
# print(class_names)
num_classes = len(image_datatsets["train"].classes)



# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         results = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             results.append(correct_k.mul_(100.0 / batch_size))
#         return results


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

# def eval_model(model,val_loader, criterion, device ="cuda"):


# 	out_loss_val = 0
# 	predicted = []
# 	ground_truth = []

# 	# acc1 = 0
# 	# acc5 = 0

# 	model.eval()
# 	with torch.no_grad():
# 		for batch_index, (data, target) in enumerate(val_loader):
# 			data = data.to(device = device)
# 			target = target.to(device= device)
# 			scores= model(data)

# 			loss = F.cross_entropy(scores, target)

# 			out_loss_val += loss.item()
# 			# print(batch_index)
# 			_, predictiones = scores.max(1)
# 			pred = predictiones.cpu().detach().numpy()
# 			target =target.cpu().detach().numpy()
# 			# print(pred)
# 			predicted += list(pred)
# 			ground_truth += list(target)

# 		accuracy = accuracy_score(ground_truth, predicted)*100
# 		return out_loss_val, accuracy


def train(model,num_epochs,train_loader,val_loader,optimizer,criterion, model_name = "ResNet_18_tiny", model_path = "/ssd_scratch/cvit/shaon/teacher_model",loss_path= "teacher_model_loss", device = "cuda"):
	train_loss = {}
	val_loss =  {}
	dur = []
	best_vaild_loss = float('inf')

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
  			if batch_index % 50 == 0:
  				print(f"TrainBatchLoss: {batch_index}\t loss :{out_loss/(batch_index+1)}",flush = True)

  	




		train_loss[epoch+1] = (out_loss/len(train_loader))


		# out_loss_val, accuracy_val = eval_model(model, val_loader,criterion,device)
 


		# val_loss[epoch+1] = (out_loss_val/len(val_loader))
		# current_val_loss = (out_loss_val/len(val_loader))

		dur.append(time.time() - t0)
		curr_lr = optimizer.param_groups[0]['lr']

		print(f'Epoch {epoch+1} \t Training Loss: {out_loss / len(train_loader)} \t LR:{curr_lr} \t Time(s):{np.mean(dur)}', flush = True)





		# if current_val_loss < best_vaild_loss:
		# 	best_vaild_loss = current_val_loss
		# 	print(f"\nSaving the best model for epoch:{epoch +1}\n", flush = True)

		torch.save(model.state_dict(),model_path+"/"+model_name +"_" +str(num_epochs))


	with open (loss_path+ "/train_loss"+"_"+model_name+"_"+str(num_epochs)+".pkl","wb") as file:
		dump(train_loss, file)
	# with open(loss_path + "/val_loss"+"_"+model_name+"_"+str(num_epochs)+".pkl","wb")as file:
	# 	dump(val_loss, file)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#initialize the pretrained model 
teacher_model = models.resnet18(pretrained = True)
# print(teacher_model)
for param in teacher_model.parameters():
	param.requires_grad =  False

num_ftrs = teacher_model.fc.in_features

teacher_model.fc = nn.Linear(num_ftrs, num_classes)


teacher_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer, _ = getOptimizer(teacher_model, 0.001, "Adam")
num_epochs = 1000
train(teacher_model, num_epochs, train_dataloaders, val_dataloaders, optimizer, criterion)









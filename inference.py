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
import torchvision.models as models
# from teacher_model_18 import teacher_model
from sklearn.metrics import *



data_transforms = {

	"new_val": transforms.Compose([transforms.ToTensor(), transforms.Resize(224, antialias = True)]),
	# "test":transforms.Compose([transforms.ToTensor()])

}

data_dir = "/ssd_scratch/cvit/shaon/imageNet-tiny/tiny-imagenet-200"

image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
					for x in ["new_val"]}


batch_size = 100
# train_dataloaders = DataLoader(image_datasets["train"], batch_size = batch_size, shuffle = "True", num_workers = 8)
test_dataloaders = DataLoader(image_datasets["new_val"], batch_size = batch_size, shuffle = "False", num_workers = 8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#initialize the pretrained model 
teacher_model = models.resnet152()
# # print(teacher_model)
# # for param in teacher_model.parameters():
# # 	param.requires_grad =  False

num_ftrs = teacher_model.fc.in_features

teacher_model.fc = nn.Linear(num_ftrs, 200)
saved_model_path = "/home2/shaon/smai_project/vit_distillation/teacher_model/ResNet_152_tiny_1000"
teacher_model.load_state_dict(torch.load(saved_model_path))

teacher_model.to(device)

predicted = []
ground_truth= []
accuracy = 0
precision = 0
recall = 0
f1 = 0

teacher_model.eval()

with torch.no_grad():
	print(f"Taking inference on :{os.path.basename(saved_model_path)} model ")
	for batch_index , (data, gt) in enumerate(test_dataloaders):
		data = data.to(device = device)
		gt = gt.to(device = device)
		# print(data.shape)
		# print(gt.shape)
		# print(gt)

		scores = teacher_model(data)
		_, predictiones = scores.max(1)
		pred = predictiones.cpu().detach().numpy()
		gt = gt.cpu().detach().numpy()

		predicted += list(pred)
		ground_truth += list(gt)
		accuracy = accuracy_score(list(gt), list(pred)) * 100

		if batch_index % 10 ==0:
			print(f"batch_index: {batch_index} \t Acc@1_test : {accuracy}")

	total_accuracy = accuracy_score(ground_truth, predicted)*100
	f1 = f1_score(ground_truth, predicted, average = "micro" )*100

	print(f"Accuracy over total test set: {total_accuracy}\t F1 Score : {f1}")

























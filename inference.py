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
from sklearn.metrics import top_k_accuracy_score
import numpy as np
torch.manual_seed(123)

transforms = transforms.Compose([transforms.ToTensor(), 
									transforms.Resize(224,antialias = True),

									transforms.Normalize((0.2675, 0.2565, 0.2761),(0.5071, 0.4867, 0.4408))])




trainset =datasets.CIFAR100(root = "/ssd_scratch/cvit/shaon", train = True, transform =transforms ,download = False)
testset = datasets.CIFAR100(root = "/ssd_scratch/cvit/shaon", train = False, transform = transforms, download = False)

batch_size = 100
train_dataloaders = DataLoader(trainset, batch_size = batch_size, shuffle = "True", num_workers = 8)
test_dataloaders = DataLoader(testset, batch_size = batch_size, shuffle = "False", num_workers = 8)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#initialize the pretrained model 
teacher_model = models.resnet50()
# # print(teacher_model)
# # for param in teacher_model.parameters():
# # 	param.requires_grad =  False

num_ftrs = teacher_model.fc.in_features

teacher_model.fc = nn.Linear(num_ftrs, 100)
# saved_model_path = "/ssd_scratch/cvit/shaon/teacher_model_resnet18_preFalse/ResNet_18_tiny_preFalse_100"
# saved_model_path = "/ssd_scratch/cvit/shaon/teacher_model_regnetv2/RegNet_y_8gf_cifar100_20"
saved_model_path = "/ssd_scratch/cvit/shaon/teacher_model_resnet50v2/ResNet50_cifar100_99"
teacher_model.load_state_dict(torch.load(saved_model_path))

teacher_model.to(device)

predicted = []
ground_truth= []
accuracy = 0
precision = 0
recall = 0
f1 = 0

teacher_model.eval()


# def accuracy(output, target, topk=(1,)):
# 	"""Computes the precision@k for the specified values of k"""
# 	maxk = max(topk)
# 	batch_size = target.size(0)

# 	_, pred = output.topk(maxk, 1, True, True)
# 	pred = pred.t()
# 	correct = pred.eq(target.reshape(1, -1).expand_as(pred))

# 	res = []
# 	for k in topk:
# 	    correct_k = correct[:k].reshape(-1).float().sum(0)
# 	    res.append(correct_k.mul_(100.0 / batch_size))
# 	return res


predicted = []
ground_truth = []

with torch.no_grad():
	print(f"Taking inference on :{os.path.basename(saved_model_path)} model ")
	for batch_index , (data, gt) in enumerate(test_dataloaders):
		data = data.to(device = device)
		gt = gt.to(device = device)


		scores = teacher_model(data)

		scores = F.softmax(scores, dim = 1)
		# print(scores.shape)
		# print(gt.shape)
		# print(scores)
		# print(gt)
		# pred = torch.argmax(scores, dim = 1)
		# pred = scores
		scores = scores.cpu().detach().numpy()
		gt = gt.cpu().detach().numpy()

		print(scores.shape)
		print(gt.shape)
		# print(gt)
		# print(pred.shape)

		labels = np.arange(0,100)
		top5 = top_k_accuracy_score(gt,scores, k=5, labels = labels)*100
		print(top5)




		# predicted += list(scores)
		# ground_truth += list(gt)


		# print(scores[0])
		# scores = scores[0].numpy()

		# gt = np.array(gt)

		# print(scores)
		# print(gt[0])

		# gt = gt.cpu().detach().numpy()
		# print(scores)
		# print(gt)
		# top1 = top_k_accuracy_score(ground_truth,predicted, k=1)
		# print(top1)




		break
		predictions = torch.argmax(scores, dim = 1)

		# print(scores.shape)

		pred = predictions.cpu().detach().numpy()
		gt = gt.cpu().detach().numpy()

		predicted += list(pred)
		ground_truth += list(gt)

	# total_accuracy = accuracy_score(ground_truth, predicted)*100
	# print(total_accuracy)





		# print(scores)
		# print(gt)

		# top1 = top_k_accuracy_score(gt, scores, k = 2)
		# top5 = top_k_accuracy_score(gt, scores, k = 5)

		# top = accuracy(scores, gt, topk= (1, 5))
		# # #top5 = accuracy(scores, gt, topk = (5,))

		# top1 = top[0].cpu().detach().numpy()
		# top5 = top[1].cpu().detach().numpy()
		

# 		# print(top1)
# 		# print(top5)

# 		# break
		# print(f" batch_index : {batch_index} \t top1_acc:{top1} \t top5_acc : {top5}")


		# print(scores.shape)
		# _, predictiones = scores.max(dim = 1)
		# predictions = torch.argmax(scores, dim = 1)
		# # print(predictions)
		# pred = predictions.cpu().detach().numpy()
		# gt = gt.cpu().detach().numpy()

		# predicted += list(pred)
		# ground_truth += list(gt)
		# accuracy = accuracy_score(list(gt), list(pred)) * 100

		# if batch_index % 100 ==0:
		# 	print(f"batch_index: {batch_index} \t Acc@1_test : {accuracy}")
	# print("pred:", predicted)
	# print("gt:", ground_truth)
	# total_accuracy = accuracy_score(ground_truth, predicted)*100
	# f1 = f1_score(ground_truth, predicted, average = "micro" )*100

	# print(f"Accuracy over total test set: {total_accuracy}\t F1 Score : {f1}")


























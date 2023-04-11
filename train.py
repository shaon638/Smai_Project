import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import timm
import torch.nn.functional as F
import os
import pycls.core.builders as builders

# def train_transforms():
# 	return transforms.Compose([transforms.ToTensor()])
# def val_transforms():
# 	return transforms.Compose([transforms.ToTensor()])

# def test_transforms()
data_transforms = {
	"train": transforms.Compose([transforms.ToTensor(), transforms.Resize(224)]),
	"val": transforms.Compose([transforms.ToTensor(), transforms.Resize(224)]),
	"test":transforms.Compose([transforms.ToTensor(), transforms.Resize(224)])
}

data_dir = "/ssd_scratch/cvit/shaon/imageNet-tiny/tiny-imagenet-200"

image_datatsets = {x : datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
					for x in ["train", "val", "test"]}

# print(image_datatsets)
batch_size = 100

train_dataloaders = DataLoader(image_datatsets["train"], batch_size = batch_size, shuffle = "True", num_workers = 2)
val_dataloaders = DataLoader(image_datatsets["val"], batch_size = batch_size, shuffle = "False", num_workers = 2)

test_dataloaders = DataLoader(image_datatsets["test"], batch_size = batch_size, shuffle = "False", num_workers = 2)

dataset_sizes = {x : len(image_datatsets[x]) for x in ["train", "val", "test"]}
# print(dataset_sizes)
# print(class_names)
num_classes = len(image_datatsets["train"].classes)
# print(class_names)

#load pretrained model as a teacher ResNet_Y_8GF model
# model_teacher = timm.create_model('resnet50', pretrained=True)
# model_teacher = timm.create_model('regnety_8gf', pretrained=True)
# model_teacher = timm.create_model('regnetx_8gf', pretrained=True)
# model_teacher = builders.build_model(arch="regnety_8gf",num_classes=1000,pretrained=True)
# model_teacher = torch.hub.load('facebookresearch/RegNet', 'regnet_y_8gf')
# model_teacher = models.regnetx_002(pretrained = True)

model_teacher = models.resnet152(pretrained = True)
in_features_teacher = model_teacher.fc.in_features
print(in_features_teacher)
model_teacher.fc = nn.Linear(in_features_teacher, num_classes)

# model_teacher = models.RegNet_y_8gf(pretrained = True)




#create a  cnn network AlexNet:-

class StudentNet(nn.Module):
    def __init__(self, num_classes=200):
        super(StudentNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0), #3x224x224 --> 96x55x55
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)) # 96x55x55 --> 96x27x27
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2), # 96x27x27 --> 256x27x27
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)) # 256x27x27 --> 256x13x13

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*13*13, 4096),
            nn.ReLU())

        self.fc1= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)

        return out

#set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

teacher = model_teacher.to(device)
teacher_size = count_params(teacher)
# print(teacher_size)

student = StudentNet(num_classes).to(device)
student_size = count_params(student)
# print(student_size)

def distillation(gt, teacher_output, student_output, T):
	ce_loss = F.cross_entropy(student_outputs, gt)
	distill_loss = F.kl_div(F.log_softmax(student_output / T, dim = 1), F.softmax(teacher_output / T, dim = 1))

	return ce_loss, distill_loss

def eval_model(model,val_loader, criterion, model_type = "MLP", device ="cpu"):

  # model.load_state_dict(torch.load(model_path)).to(device)
  out_loss = 0
  # out_loss_distill_val = 0
  model.eval()

  for batch_index, (data, target) in enumerate(val_loader):
    data = data.to(device = device)
    target = target.to(device= device)


    with torch.no_grad():

      scores= model(data)

    loss = F.cross_entropy(scores, target)

    out_loss_test += loss.item()
    encoded_img_feature_testSet.append(encoded_img_testSet.cpu().detach().numpy())
    target_test_set.append(target.detach().numpy())

  return out_loss_test, data, recon_img, encoded_img_feature_testSet, target_test_set

# def train(model_student,model_teacher,num_epochs,train_loader,val_loader, optimizer,criterion,T = 20,alpha =0.7, model_name = "Adam", model_path = "models",loss_path= "losses" ,is_lrSchedular = False, type_lrSch = None, device = "cpu"):
  
#   model_student.train()
#   model_teacher.eval()
#   train_ce_loss = {}
#   # val_ce_loss =  {}
#   train_dist_loss ={}
#   # val_dist_loss = {}
#   dur = []
#   if is_lrSchedular:
#     if type_lrSch == "ExponentialLR":
#       scheduler = ExponentialLR(optimizer, gamma=0.9)
#     elif type_lrSch == "LinearLR":
#       scheduler = LinearLR(optimizer, start_factor=0.1,end_factor=0.8, total_iters=20)
#   for epoch in range(num_epochs):
# 	t0 = time.time()
    
# 	out_loss_ce_train = 0
# 	out_loss_distill_train = 0
   
# 	model_student.train()
# 	model_teacher.eval()
#     for batch_index, (data, target) in enumerate(train_loader):
#       data = data.to(device = device)
#       target = target.to(device= device)

#       #shape of the data(batch_size, 3, 224, 224)

#       #flatten the shape into single dimension
#       # data = data.reshape(data.shape[0], -1)
#       # print(data.shape)
#       # break

#       #forward 
#       scores_student = model_student(data)
#       scores_teacher = model_teacher(data)
#       ce_loss, distill_loss = distillation(target,scores_teacher, scores_student, T)
#       #soft distillation loss ....
#       loss = (1-alpha)* ce_loss + (alpha * T ** 2)*distill_loss


#       # loss = criterion(scores, target)

#       #backword
#       optimizer.zero_grad()
#       loss.backward()

#       #gradient descent
#       optimizer.step()

#       out_ce_loss += ce_loss.cpu().item()
#     train_ce_loss[epoch+1] = (out_loss/len(train_loader))

    
#     out_loss_test = 0
#     model.eval()
    
#     for batch_index, (data, target) in enumerate(val_loader):
#       data = data.to(device = device)
#       target = target.to(device= device)
#       data = data.reshape(data.shape[0], -1)
#       with torch.no_grad():

#         scores = model(data)

#       loss = criterion(scores, target)

#       out_loss_test += loss.item()
#     test_loss[epoch+1] = (out_loss_test/len(val_loader))
    
#     dur.append(time.time() - t0)
#     curr_lr = optimizer.param_groups[0]['lr']
#     print(f'Epoch {epoch+1} \t Training Loss: {out_loss / len(train_loader)} \t Val Loss: {out_loss_test / len(val_loader)} \t LR:{curr_lr} \t Time(s):{np.mean(dur)}')
#     if is_lrSchedular:
#       scheduler.step(out_loss_test/len(val_loader))
#   if is_lrSchedular:
#     torch.save(model.state_dict(),model_path+"/"+model_name+"_"+type_lrSch+"_"+str(num_epochs))
#     with open (loss_path+ "/train_loss"+"_"+model_name+"_"+type_lrSch+"_"+str(num_epochs)+".pkl","wb") as file:
#       dump(train_loss, file)
#     with open(loss_path + "/val_loss"+"_"+model_name+"_"+type_lrSch+"_"+str(num_epochs)+".pkl","wb")as file:
#       dump(test_loss, file)
#   else:
#     torch.save(model.state_dict(),model_path+"/"+model_name+"_"+str(num_epochs))
#     with open (loss_path+ "/train_loss"+"_"+model_name+"_"+str(num_epochs)+".pkl","wb") as file:
#       dump(train_loss, file)
#     with open(loss_path + "/val_loss"+"_"+model_name+"_"+str(num_epochs)+".pkl","wb")as file:
#       dump(test_loss, file)













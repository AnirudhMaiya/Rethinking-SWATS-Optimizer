import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import torchvision
from torchvision.transforms import transforms
from Resnet import ResNet,BasicBlock
from SwatsVanillaGlobal import SwatsVanillaGlobal
import numpy as np
import math
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

batch_is = 128
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.247, 0.243, 0.261])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.247, 0.243, 0.261])
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_is,
                                          shuffle=True, num_workers=1,pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_is,
                                         shuffle=False, num_workers=1,pin_memory=True)

def accuracy(y_hat,y_true):
  y_hat = F.softmax(y_hat,dim = 1)
  _, predicted = torch.max(y_hat, 1)
  total_correct = (predicted.reshape(-1,1) == y_true.reshape(-1,1)).sum().item()
  return total_correct

def train(model,epochs,loader):
  model.train()
  correct = 0
  cc = 0
  loss_list = []
  for i,j in loader:
    inputs,labels = i.to(device),j.to(device)
    opt.zero_grad()
    outputs = model(inputs)
    loss_is = loss(outputs,labels)
    loss_is.backward()
    opt.step()
    loss_list.append(loss_is.item())
    correct = correct + accuracy(outputs,labels)
  
  print("[%d/%d] Training Accuracy : %f"%(epochs,total_epochs, (correct/len(loader.dataset)) * 100))
  return sum(loss_list)/len(loss_list),(correct/len(loader.dataset)) * 100

def test(model,epochs,loader):
  model.eval()
  correct = 0
  with torch.no_grad():
    for i,j in loader:
      inputs,labels = i.to(device),j.to(device)
      outputs = model(inputs)
      correct = correct + accuracy(outputs,labels)
    print("[%d/%d] Test Accuracy : %f"%(epochs,total_epochs,(correct/len(loader.dataset))*100))
    print('---------------------------------------------------------------------')
  return (correct/len(loader.dataset)) * 100

dtype = torch.cuda.FloatTensor
torch.manual_seed(52)
net = ResNet(BasicBlock,[2,2,2,2]).to(device)
opt = SwatsVanillaGlobal(net.parameters(),lr = 0.001)
loss = nn.CrossEntropyLoss().type(dtype)

def adjust_lr(opt,epochs):
  base_lr = 0.001
  if epochs >= 100:
    for ui in opt.param_groups:
      ui['lr_decay'] = 10

total_epochs = 200
train_loss = []
train_acc = []
test_acc = []
for s in range(1,total_epochs + 1):
  adjust_lr(opt,s)
  a,b = train(net,s,trainloader)
  c = test(net,s,testloader)
  train_loss.append(a)
  train_acc.append(b)
  test_acc.append(c)
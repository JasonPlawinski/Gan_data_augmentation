import math
import pickle
import torch
import torch.nn as nn
import numpy as np
import math, random
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import matplotlib
import matplotlib.pyplot as plt

batch_size = 100

mnist_train = datasets.MNIST('./MNIST/', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = datasets.MNIST('/MNIST/', train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=1,shuffle=True)

def Corrupt_image(batch,pCType,Param = [5, 5, 5, 0.2, 0.05, 3, 0]):
    ''' Uniform -- Gaussian -- Masking -- Dropout -- Saturation -- Multiplicative -- SuperResoltion
    pCType is the ratio of image for a given type
    Param is the parameter for a given noise'''
    return 0

def Corrupt_Gaussian(Image, sigma):
    GN = Image + torch.normal(torch.zeros(Image.size()))*sigma
    GN[GN<0] = 0
    GN[GN>1] = 1
    return GN

def Corrupt_Uniform(Image, u):
    UN = Image + (torch.rand(Image.size()) - torch.ones(Image.size())*0.5)*2*u
    UN[UN<0] = 0
    UN[UN>1] = 1
    return UN

def Corrupt_SaltPepper(Image, p):
    prob = p/2
    MaskSalt = torch.rand(Image.size())
    MaskSalt[MaskSalt>(1-prob)] = 1
    MaskSalt[MaskSalt<(1-prob)] = 0
    
    MaskPepper = torch.rand(Image.size())
    MaskPepper[MaskPepper<(1-prob)] = 0
    MaskPepper[MaskPepper>(1-prob)] = -2
    
    CorrSP = MaskSalt + Image + MaskPepper
    CorrSP[CorrSP<0] = 0
    CorrSP[CorrSP>1] = 1
    return CorrSP

def Corrupt_Mult(Image, sigma):
    MultN = Image*( 1 + torch.normal(torch.zeros(Image.size()))*sigma)
    MultN[MultN<0] = 0
    MultN[MultN>1] = 1
    return MultN

def Corrupt_Mask(Image, p, l=[3,5], corruptValue = 1):
    ImReshaped = Image.view(-1,1,28,28)
    S = ImReshaped.size()
    Pos = torch.rand(ImReshaped.size())
    Pos[Pos>(1-p)] = 1
    Pos[Pos<(1-p)] = 0    
    padLateral = int(np.floor(l[1]*0.5))
    padHeight = int(np.floor(l[0]*0.5))
    sizeKernel = [1, 1, int(l[0]), int(l[1])]
    Kernel = torch.ones(sizeKernel).float()
    Pad = torch.nn.ZeroPad2d([padLateral,padLateral,padHeight,padHeight])
    ConvOp = nn.Conv2d(in_channels=S[1], out_channels=S[1], kernel_size=3, stride =1, padding=0, bias= False)
    ConvOp.weight.data = Kernel
    Masks = ConvOp(Pad(Variable(Pos, requires_grad = False))).data
    MaskNoise = Image
    MaskNoise[Masks > 0] = corruptValue
    return MaskNoise

def DownSampling(Image):
    DS = nn.MaxPool2d(2)
    Bilin = nn.Upsample(scale_factor=2, mode='bilinear')
    LR = DS(Image)
    HR = Bilin(LR)
    return(HR.data)
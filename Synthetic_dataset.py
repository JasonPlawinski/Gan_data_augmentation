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
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from Class_Networks import *
from Corrupt import *

z_dim = 50

def Create_Synthetic(nsamples):
    
    #Load EncoderNet
    Generator = Conv_Decoder(z_dim)
    Generator = Generator.cuda()
    Generator.load_state_dict(torch.load("./NetworkSaves/GAN_Class_20000.pt"))
    
    Synthetic_Digits = torch.zeros([10*nsamples,28,28])
    label = torch.zeros(10*nsamples).long()
    
    index = 0
    for i in range(10):
        label_G = (torch.ones([nsamples])*i).long()
        label_hot_G = torch.zeros(len(label_G),10)

        label_hot_G[np.arange(len(label_G)), label_G] = 1.0
        label_hot_G = Variable(label_hot_G.float(), requires_grad=True).cuda()
        
        label[index*nsamples:(index+1)*nsamples] = i
        
        noise_G = Variable(torch.randn([nsamples,z_dim-10])).cuda()
        '''print(noise_G.size())
        print(label_hot_G.size())'''
        noise_G = torch.cat((noise_G , label_hot_G), 1)
        Im_fake_G = Generator(noise_G).cpu().data.view([nsamples,28,28])
        #Im_cpu = Im_fake_G.data.cpu().numpy()
        Synthetic_Digits[index*nsamples:(index+1)*nsamples,:,:] = (Im_fake_G)
        index += 1 
        
    return(Synthetic_Digits, label)

def Concatenate_Dataset(Real_Digits,Real_Label,nsamples):
    Real_Digits = torch.from_numpy(Real_Digits)
    Real_Label = torch.from_numpy(Real_Label)
    if nsamples == 0:
        return Real_Digits, Real_Label
    Fake_Digits, Fake_Label = Create_Synthetic(nsamples)
    Digits = torch.cat((Fake_Digits, Real_Digits), 0)
    Label = torch.cat((Fake_Label, Real_Label), 0)
    return Digits, Label
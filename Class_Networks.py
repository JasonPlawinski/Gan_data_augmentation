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

#Encoder
class Conv_Encoder(nn.Module):
    '''The Encoder Network is a convolutional network with a few convolutional
    The convolutions are 2D strided to reduce the size of the image. The last layer is fully connected
    Dropout is applied to all layers except the dense one
    Activation is leaky ReLU
    The output of the convolutions are normalized using Instance Normalization
    Output is not Normalized'''
    def __init__(self, z_dim):
        super(Conv_Encoder, self).__init__()
        self.Input = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1,16,3,1,0),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
        
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(16,32,3,1,0),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,32,3,2,0),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
            
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,32,3,2,0),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
            
        self.layer4 = nn.Sequential(
            nn.Conv2d(32,32,3,2,0),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
        
        self.LinLayers = nn.Sequential(
            nn.Linear(128, 100),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2),
            nn.Linear(100, z_dim))
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x, batch_size = 100):
        out = self.Input(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        S = x.size()[0]
        out = out.view([S,1,-1])
        out = out.view([S,-1])
        out = self.LinLayers(out)
        return out
    

#Encoder
class Conv_Encoder_Concat(nn.Module):
    '''The Encoder Network is a convolutional network with a few convolutional
    The convolutions are 2D strided to reduce the size of the image. The last layer is fully connected
    Dropout is applied to all layers except the dense one
    Activation is leaky ReLU
    The output of the convolutions are normalized using Instance Normalization
    Output is not Normalized'''
    def __init__(self, z_dim):
        super(Conv_Encoder_Concat, self).__init__()
        self.Input = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1,15,3,1,0),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
        
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(16,32,3,1,0),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,32,3,2,0),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
            
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,32,3,2,0),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
            
        self.layer4 = nn.Sequential(
            nn.Conv2d(32,32,3,2,0),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
        
        self.LinLayers = nn.Sequential(
            nn.Linear(128, 100),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2),
            nn.Linear(100, z_dim))
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x, y, batch_size = 100):
        out = self.Input(x)
        y2 = y.repeat(1,28*28).view(-1,1,28,28)
        out = torch.cat((out,y2),1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        S = x.size()[0]
        out = out.view([S,1,-1])
        out = out.view([S,-1])
        out = self.LinLayers(out)
        return out
    
    
class Conv_Decoder(nn.Module):
    '''The Decoder Network starts off with dense layers then followed up with 2D convolutions and Pixel shuffler to upscale the image
    The pixel shuffler use fractional convolutions and upscale an image by re arranging pixels
    from multiple channels into one larger channel.
    For example for x2 upsampling in 2D, the output of the pixel shuffler will have 4 times less channels
    Activation is Leaky RelU
    The output of a block of convolutions is normalized using Instance Normalization
    Dropout is applied to all hidden layers
    Output is Sigmoid'''
    def __init__(self, z_dim):
        super(Conv_Decoder, self).__init__()
        self.LinLayers = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 196),
            nn.LeakyReLU(0.2))
            
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.25),
            nn.Conv2d(16,16,3,1,1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16,16,3,1,1),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(2),
            
            nn.LeakyReLU(0.2),
            nn.Conv2d(4,16,3,1,1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.25))
            
        self.Output = nn.Sequential(
            nn.Conv2d(16,16,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.25),
            nn.Conv2d(16,1,3,1,1),
            nn.Sigmoid())
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            
    def forward(self, x, batch_size = 100):
        out = self.LinLayers(x)
        S = x.size()[0]
        out = out.view([S,-1,14,14])
        out = self.layer1(out)
        out = self.Output(out)
        return out
    

#Encoder
class Discriminator_Concat_layers(nn.Module):
    '''The Encoder Network is a convolutional network with a few convolutional
    The convolutions are 2D strided to reduce the size of the image. The last layer is fully connected
    Dropout is applied to all layers except the dense one
    Activation is leaky ReLU
    The output of the convolutions are normalized using Instance Normalization
    Output is not Normalized'''
    def __init__(self, z_dim):
        super(Discriminator_Concat_layers, self).__init__()
        self.Input = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1,8,3,1,0),
            nn.InstanceNorm2d(8),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
        
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(8,16,3,2,0),
            nn.Dropout(p=0.25),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(26,32,3,2,0),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
            
        self.layer3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32,32,3,1,0),
            nn.Dropout(p=0.25),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2))
            
        self.layer4 = nn.Sequential(
            nn.Conv2d(32,16,3,2,0),
            nn.InstanceNorm2d(16),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
        
        self.LinLayers = nn.Sequential(
            nn.Linear(64, 100),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2),
            nn.Linear(100, z_dim))
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x, y, batch_size = 100):
        out = self.Input(x)
        out = self.layer1(out)
        y2 = y.repeat(1,14*14).view(-1,10,14,14)
        out = torch.cat((out,y2),1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        S = x.size()[0]
        out = out.view([S,1,-1])
        out = out.view([S,-1])
        out = self.LinLayers(out)
        return out

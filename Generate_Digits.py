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
from PIL import Image
from Class_Networks import *
from Corrupt import *

batch_size = 100
z_dim = 50

def Compil_Image(ImBatch, name, digit_number_h=10, digit_number_l = 10, digit_size = [28,28]):
    TestImage = np.zeros([digit_size[0]*digit_number_h, digit_size[1]*digit_number_l])
    index = 0
    for i in range(digit_number_h):
        for j in range(digit_number_l):
            temp = ImBatch[index][0].cpu().data.numpy()
            TestImage[i*digit_size[0]:(i+1)*digit_size[0],j*digit_size[1]:(j+1)*digit_size[1]] = temp
            index +=1
    #plt.imshow(TestImage, cmap = 'gray', interpolation = 'none')
    #plt.axis('off')
    #plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi = 300)
    TestImage = Image.fromarray(np.uint8(TestImage*255))
    TestImage.save(name)
    


torch.manual_seed(2)
mnist_train = datasets.MNIST('./MNIST/', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = datasets.MNIST('/MNIST/', train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=1,shuffle=True)

#Load EncoderNet
Generator = Conv_Decoder(z_dim)
Generator = Generator.cuda()
Generator.load_state_dict(torch.load("./NetworkSaves/GAN_Class_15000.pt"))

for i in range(1):
    label_G = (torch.ones([batch_size])*i).long()
    label_G = (torch.rand([nsamples])*10).long()
    label_hot_G = torch.zeros(len(label_G),10)

    label_hot_G[np.arange(len(label_G)), label_G] = 1.0
    label_hot_G = Variable(label_hot_G.float(), requires_grad=True).cuda()
    noise_G = Variable(torch.randn([batch_size,z_dim-10])).cuda()
    noise_G = torch.cat((noise_G , label_hot_G), 1)
    Im_fake_G = Generator(noise_G)
    name = './Generated_Digits/Test_Digit_' + str(i) + '.png'
    Compil_Image(Im_fake_G, name)

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
from Synthetic_dataset import *


torch.manual_seed(10)
batch_size = 50
small_mnist_size = 500

mnist_train = datasets.MNIST('./MNIST/', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = datasets.MNIST('/MNIST/', train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=small_mnist_size,shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=1,shuffle=True)
train_full = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=50000,shuffle=False)

prob = 0.2
X_dim = 784
N = 600
N0 = 150
z_dim = 10

Encoder = Conv_Encoder(z_dim)
Encoder = Encoder.cuda()
# Set learning rates
gen_lr, reg_lr = 0.0005, 0.0008
# Set optimizators
Encoder_grad = optim.Adam(Encoder.parameters(), lr=gen_lr, weight_decay = 0.0001)
criterionCEL = nn.CrossEntropyLoss().cuda()

epochs = 5000

perm_index = np.load('./NetworkSaves/perm.npy')
perm_index = np.array(perm_index)

full_mnist = iter(train_full).next()
index = full_mnist[1][:].numpy()
index = index[perm_index]
index = index[:small_mnist_size]

full_mnist = full_mnist[0][:].numpy().reshape(50000,784)
full_mnist = full_mnist[perm_index]
small_mnist = full_mnist[:small_mnist_size]

Real_Digits = small_mnist[:small_mnist_size].reshape([small_mnist_size,28,28])
nsamples = 0
data, label_set = Concatenate_Dataset(Real_Digits, index, nsamples)
data_size = data.size()[0]
data = data.view([data_size,1,28,28])

for e in range(epochs):
    if e ==0:
        print('Start')
    for i in range(data_size//batch_size):
        loss_l = []
        X = Variable(data[i*batch_size:(i+1)*batch_size]).cuda()
        y = Encoder(X)
        
        label = label_set[i*batch_size:(i+1)*batch_size]
        
        label_hot = torch.zeros(len(label),10)
        label_hot[np.arange(len(label)),label] = 1.0
        label_hot = Variable(label_hot.long(), requires_grad=False).cuda()
        loss = criterionCEL(y, Variable(label.long(),requires_grad = False).cuda())
        Encoder_grad.zero_grad()
        loss.backward()
        Encoder_grad.step()
        loss_l.append(loss)
        perm_index = torch.from_numpy(np.random.permutation(data_size))
           
    data = data[perm_index]
    label_set = label_set[perm_index]
    
    if e%100 == 0:
        good_class = 0
        for i in range(401):
            pair = iter(test_loader).next()
            img = pair[0]
            label_test = pair[1]
            X = Variable(img).cuda()
            y = Encoder(X)
            '''label_hot_test = torch.zeros(len(label_test),10)
            label_hot_test[np.arange(len(label_test)),label_test] = 1.0
            label_hot_test = label_hot_test.numpy()'''
            res = y.data.view([10]).cpu().numpy()
            val = np.argmax(res)
            if val == label_test.numpy():
                good_class += 1
            
        print('Test accuracy', good_class/400)
        a = np.mean(np.array(loss_l)).data.cpu().numpy()
        print('Training loss', a[0])
        print()

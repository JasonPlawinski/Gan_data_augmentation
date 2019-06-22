import math
import pickle
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
import math, random
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from Net_Architecture import *
from Corrupt import *

prob = 0.2
X_dim = 784
N = 600
N0 = 150
z_dim = 2

torch.manual_seed(2)
mnist_train = datasets.MNIST('./MNIST/', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = datasets.MNIST('/MNIST/', train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=1,shuffle=True)

#Load EncoderNet
Encoder = Conv_Encoder(z_dim)
Encoder = Encoder.cuda()
Encoder.load_state_dict(torch.load("./NetworkSaves/Conv_2.pt"))

ValSetSize = 2000

Code = np.zeros([ValSetSize, z_dim])
Array_classes = np.zeros([ValSetSize, 1],dtype = np.int)

for i in range(ValSetSize):
    pair = iter(test_loader).next()
    testimg = pair[0]
    label = pair[1]
    testimg = Variable(testimg.view([1,1,28,28])).cuda()
    coor = Encoder(testimg)
    coor = coor[0][0].data.cpu().numpy()
    label = label.cpu().numpy()[0]
    Code[i, :] = coor[:]
    Array_classes[i] = label

n_classes = 10
n_clusters = 12
cluster_pred = KMeans(n_clusters).fit_predict(Code)


#computing clusters purity
purity = np.zeros(n_clusters)

ClusterArray = np.zeros([n_clusters, n_classes])
for i in range(n_clusters):
    c_classes = np.zeros(n_classes)
    temp_classes = Array_classes[cluster_pred == i]
    for j in range(n_classes):
        #check classes for each cluster
        Bool = (temp_classes == j)
        c_classes[j] = sum(Bool*1.0)
    ClusterArray[i,:] = c_classes[:]
    maxClass =  np.max(c_classes)
    if maxClass == np.sum(c_classes):
        purity[i] = 1.0
    else:
        purity[i] = maxClass/(np.sum(c_classes))

np.save('./ClusteringResults/Conv_2_Sig.txt', ClusterArray)
print(purity)
print(np.mean(purity))
    
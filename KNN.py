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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC


torch.manual_seed(49)
batch_size = 25
small_mnist_size = 500
test_size = 500

mnist_train = datasets.MNIST('./MNIST/', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = datasets.MNIST('/MNIST/', train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=small_mnist_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=test_size,shuffle=True)

small_mnist = iter(train_loader).next()
perm_index = torch.from_numpy(np.random.permutation(small_mnist_size))
small_mnist2 = [small_mnist[0][perm_index],small_mnist[1][perm_index]]

neigh1 = KNeighborsClassifier(n_neighbors=1)
neigh3 = KNeighborsClassifier(n_neighbors=3)
SVM = LinearSVC()
NL_SVM = SVC()

flat_img = small_mnist2[0].numpy().reshape([small_mnist_size,784])
label = small_mnist2[1].numpy()


neigh1.fit(flat_img,label)
neigh3.fit(flat_img,label)
SVM.fit(flat_img, label)
NL_SVM.fit(flat_img, label)

small_mnist_test = iter(test_loader).next()
perm_index = torch.from_numpy(np.random.permutation(test_size))
small_mnist_test2 = [small_mnist_test[0][perm_index],small_mnist_test[1][perm_index]]

flat_img_test = small_mnist_test2[0].numpy().reshape([test_size,784])
label_test = small_mnist_test2[1].numpy()

pred = neigh1.predict(flat_img_test)
a = (pred==label_test)*1.0
print(np.sum(a)/len(a))

pred = neigh3.predict(flat_img_test)
a = (pred==label_test)*1.0
print(np.sum(a)/len(a))

pred_SVM = SVM.predict(flat_img_test)
a = (pred_SVM==label_test)*1.0
print(np.sum(a)/len(a))

NL_pred_SVM = NL_SVM.predict(flat_img_test)
a = (NL_pred_SVM==label_test)*1.0
print(np.sum(a)/len(a))



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
from matplotlib.lines import Line2D
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from Class_Networks import *
from Corrupt import *

batch_size = 100
z_dim = 50
small_mnist_size = 500

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

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=small_mnist_size,shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=1,shuffle=False)
train_full = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=50000,shuffle=False)

#Load EncoderNet
Generator = Conv_Decoder(z_dim)
Generator = Generator.cuda()
Generator.load_state_dict(torch.load("./NetworkSaves/GAN_Class_20000.pt"))

Synthetic_Digits = np.zeros([10, batch_size,28,28])
for i in range(10):
    label_G = (torch.ones([batch_size])*i).long()
    label_hot_G = torch.zeros(len(label_G),10)

    label_hot_G[np.arange(len(label_G)), label_G] = 1.0
    label_hot_G = Variable(label_hot_G.float(), requires_grad=True).cuda()
    noise_G = Variable(torch.randn([batch_size,z_dim-10])).cuda()
    noise_G = torch.cat((noise_G , label_hot_G), 1)
    Im_fake_G = Generator(noise_G)
    Im_cpu = Im_fake_G.data.cpu().numpy().reshape([batch_size,28,28])
    Synthetic_Digits[i,:,:,:] = (Im_cpu)
    name = './Generated_Digits/Init_200_Digit_' + str(i) + '.png'
    Compil_Image(Im_fake_G, name)
    
Synthetic_Digits = Synthetic_Digits.reshape([10*batch_size,28,28])   
perm_index = np.load('./NetworkSaves/perm.npy')
print(perm_index.shape)
print(type(perm_index))
print(type(perm_index[0]))
perm_index = np.array(perm_index)

full_mnist = iter(train_full).next()
index = full_mnist[1][:].numpy()
index = index[perm_index]
index = index[:small_mnist_size]

full_mnist = full_mnist[0][:].numpy().reshape(50000,784)
full_mnist = full_mnist[perm_index]
small_mnist = full_mnist[:small_mnist_size]

Real_Digits = small_mnist[:small_mnist_size].reshape([small_mnist_size,28,28])

data_tot = np.concatenate((Synthetic_Digits.reshape([10*batch_size, 784]), Real_Digits.reshape([small_mnist_size,784])), axis = 0)
Mnist_Synthetic = np.concatenate((data_tot,full_mnist[:4000,:]), axis = 0)

Embed = TSNE(n_components=2)
X = Embed.fit_transform(Mnist_Synthetic)

#X = pca.transform(data_tot)

e = 1/255.0
color2 = [(24*e, 96*e, 143*e), (206*e, 98*e, 0*e), (35*e, 126*e, 35*e), (171*e, 31*e, 31*e), (121*e, 72*e, 166*e), (111*e, 68*e, 60*e),
          (216*e, 64*e, 171*e), (106*e, 106*e, 106*e), (153*e, 153*e, 28*e), (19*e, 153*e, 168*e)]  

for i in range(10):
    curr_color = 'C'+ str(i)
    x_temp = X[batch_size*i:batch_size*(i+1),0]
    y_temp = X[batch_size*i:batch_size*(i+1),1]
    plt.scatter(x_temp, y_temp, s = 8,color = color2[i], marker = '*', alpha = 0.8)

      
for i in range(10):
    temp = X[batch_size*10:batch_size*10+small_mnist_size]
    curr_color = 'C'+ str(i)
    selection = (index == i)
    x_temp = temp[selection,0]
    y_temp = temp[selection,1]
    plt.scatter(x_temp, y_temp, s = 8,color = curr_color, alpha = 0.5)
    
    
custom_legends = [Line2D([0], [0], marker='o', alpha = 1 ,color='w', label='Real',
                          markerfacecolor='black', markersize=5),
                 Line2D([0], [0], marker='*', alpha = 1 ,color='w', label='Synthetic',
                          markerfacecolor='black', markersize=8),
                   Line2D([0], [0], marker='o', alpha = 1 ,color='w', label='0',
                          markerfacecolor='C0', markersize=5),
                 Line2D([0], [0], marker='o', alpha = 1 ,color='w', label='1',
                          markerfacecolor='C1', markersize=5),
                 Line2D([0], [0], marker='o', alpha = 1 ,color='w', label='2',
                          markerfacecolor='C2', markersize=5),
                 Line2D([0], [0], marker='o', alpha = 1 ,color='w', label='3',
                          markerfacecolor='C3', markersize=5),
                 Line2D([0], [0], marker='o', alpha = 1 ,color='w', label='4',
                          markerfacecolor='C4', markersize=5),
                 Line2D([0], [0], marker='o', alpha = 1 ,color='w', label='5',
                          markerfacecolor='C5', markersize=5),
                 Line2D([0], [0], marker='o', alpha = 1 ,color='w', label='6',
                          markerfacecolor='C6', markersize=5),
                 Line2D([0], [0], marker='o', alpha = 1 ,color='w', label='7',
                          markerfacecolor='C7', markersize=5),
                 Line2D([0], [0], marker='o', alpha = 1 ,color='w', label='8',
                          markerfacecolor='C8', markersize=5),
                 Line2D([0], [0], marker='o', alpha = 1 ,color='w', label='9',
                          markerfacecolor='C9', markersize=5),]
plt.legend(handles = custom_legends, fontsize = 6)
plt.title('t-SNE embedding of MNIST digits')
plt.savefig('./Images/tsne0.png',dpi = 300)

"""31 119 180 => 24 96 143
255  127 14 => 206 98 0
44 160 44 => 35 126 35
214 39 40 => 171 31 31
148 130 189 = > 121 72 166
140 86 75 => 111 68 60
227 119 194 => 216 67 171 
127 127 127 = > 106 106 106
188 189 34 = > 153 153 28
23 190 207 => 19 153 168"""

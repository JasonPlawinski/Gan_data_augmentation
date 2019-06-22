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

torch.manual_seed(10)
batch_size = 100
small_mnist_size = 500

mnist_train = datasets.MNIST('./MNIST/', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = datasets.MNIST('/MNIST/', train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=50000,shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=1,shuffle=False)

prob = 0.2
X_dim = 784
N = 600
N0 = 150
z_dim = 50

Generator = Conv_Decoder(z_dim)
Generator = Generator.cuda()

Discriminator = Discriminator_Concat_layers(1)
Discriminator = Discriminator.cuda()
# Set learning rates
gen_lr, reg_lr = 0.0001, 0.0008
# Set optimizators
Generator_grad = optim.Adam(Generator.parameters(), lr=gen_lr, weight_decay = 0.00001)
Discriminator_grad = optim.Adam(Discriminator.parameters(), lr=gen_lr, weight_decay = 0.00001)

#parameters
Generator_parameters = filter(lambda p: p.requires_grad, Generator.parameters())
params_G = sum([np.prod(p.size()) for p in Generator_parameters])
Discriminator_parameters = filter(lambda p: p.requires_grad, Discriminator.parameters())
params_D = sum([np.prod(p.size()) for p in Discriminator_parameters])
print("Learnable parameters in Generator :", params_G)
print("Learnable parameters in Discriminator :", params_D)

criterionBCE = nn.BCEWithLogitsLoss().cuda()

epochs = 50001

full_mnist = iter(train_loader).next()
'''perm_index = np.random.permutation(50000)
np.save('./NetworkSaves/perm.npy', perm_index)'''
perm_index = np.load('./NetworkSaves/perm.npy')


perm_index = torch.from_numpy(perm_index)
small_mnist = full_mnist[:][:small_mnist_size]
small_mnist2 = [small_mnist[0][perm_index],small_mnist[1][perm_index]]
del full_mnist

k=0
for e in range(epochs):
    perm_index = torch.from_numpy(np.random.permutation(small_mnist_size))
    small_mnist2 = [small_mnist2[0][perm_index],small_mnist2[1][perm_index]]

    loss_lg = []
    loss_ld = []
    if e ==0:
        print('Start')
    for i in range(small_mnist_size//batch_size):
        batch_index = 0
        TrainFake = True
        Im_real = Variable(small_mnist2[0][i*batch_size:(i+1)*batch_size]).cuda()
        
        label_real = small_mnist2[1][i*batch_size:(i+1)*batch_size]
        label_hot_real = torch.zeros(len(label_real),10)
        label_hot_real[np.arange(len(label_real)),label_real] = 1.0
        label_hot_real = Variable(label_hot_real.float()).cuda()
        
        small_mnist2[1][i*batch_size:(i+1)*batch_size]
        
        label_D = (torch.rand([batch_size])*10).long()
        if torch.sum(label_D >= 10) > 0:
            label_D = (torch.rand([batch_size])*10).long()
            print(e)
        label_hot_D = torch.zeros(len(label_D),10)
        
        label_hot_D[np.arange(len(label_D)), label_D] = 1.0
        label_hot_D = Variable(label_hot_D.float(), requires_grad=True).cuda()
        noise_D = Variable(torch.randn([batch_size,z_dim-10])).cuda()
        noise_D = torch.cat((noise_D , label_hot_D), 1)
        Im_fake_D = Generator(noise_D)
        
        label_G = (torch.rand([batch_size])*10).long()
        if torch.sum(label_G >= 10) > 0:
            label_G = (torch.rand([batch_size])*10).long()
            print(e)
        label_hot_G = torch.zeros(len(label_G),10)
        
        label_hot_G[np.arange(len(label_G)), label_G] = 1.0
        label_hot_G = Variable(label_hot_G.float(), requires_grad=True).cuda()
        noise_G = Variable(torch.randn([batch_size,z_dim-10])).cuda()
        noise_G = torch.cat((noise_G , label_hot_G), 1)
        Im_fake_G = Generator(noise_G)
        
        #label = small_mnist2[1][i*batch_size:(i+1)*batch_size]
        #label_hot = torch.zeros(len(label),10)
        #label_hot[np.arange(len(label)),label] = 1.0
        #label_hot = Variable(label_hot.long(), requires_grad=False).cuda()
 
        prediction_fake_G = Discriminator(Im_fake_G, label_hot_G)
        prediction_fake_D = Discriminator(Im_fake_D.detach(), label_hot_D)
        prediction_real = Discriminator(Im_real, label_hot_real)

        
        #Logits
        logits0 = Variable(torch.ones(prediction_fake_G.size()).cuda(), requires_grad = False)
        logits1 = Variable(torch.zeros(prediction_fake_G.size()).cuda(), requires_grad = False)
        
        loss_d = criterionBCE(prediction_fake_D, logits0) + criterionBCE(prediction_real, logits1) 
        loss_g = criterionBCE(prediction_fake_G, logits1)
        
        Discriminator_grad.zero_grad()
        loss_d.backward()
        Discriminator_grad.step()
        
        Generator_grad.zero_grad()
        loss_g.backward()
        Generator_grad.step()
        loss_lg.append(loss_g.data[0])
        loss_ld.append(loss_d.data[0]) 
    
    if e%100 == 0:
        if (e%2500 == 0 and not(e==0)):
            SaveName = "./NetworkSaves/GAN_Class_" + str(e) + ".pt"
            torch.save(Generator.state_dict(), SaveName)
        print(e)
        print('loss G %.6f' % np.mean(np.array(loss_lg)))
        print('loss D %.6f'% np.mean(np.array(loss_ld)))
        print('')
        plt.imshow(Im_fake_D[0][0].data.cpu().numpy(), cmap = 'gray')
        label = label_D[0]
        
        '''name = str(e)
        if e< 10:
            name = '0'+name
        if e< 100:
            name = '0'+name
        if e< 1000:
            name = '0'+name
        if e< 10000:
            name = '0'+name'''
        name = str(k)
        if k< 10:
            name = '0'+name
        if k< 100:
            name = '0'+name
        plt.savefig('./Synthetic/'+name+'_'+str(label)+'.png')
        k += 1
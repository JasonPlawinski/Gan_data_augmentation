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
from Net_Architecture import *
from Corrupt import *

batch_size = 100

mnist_train = datasets.MNIST('./MNIST/', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = datasets.MNIST('/MNIST/', train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=1,shuffle=True)

prob = 0.2
z_dim = 2

torch.manual_seed(10)
Encoder = Conv_Encoder(z_dim)
Decoder = Conv_Decoder(z_dim)     # Encoder/Decoder
Discriminator = Conv_Discriminator(z_dim)

Encoder = Encoder.cuda()
Decoder = Decoder.cuda()
Discriminator = Discriminator.cuda()


# Set learning rates
gen_lr, reg_lr = 0.0001, 0.0008
# Set optimizators
Encoder_grad = optim.Adam(Encoder.parameters(), lr=gen_lr)
Decoder_grad = optim.Adam(Decoder.parameters(), lr=gen_lr)
Discriminator_grad = optim.Adam(Discriminator.parameters(), lr=gen_lr)

criterionL2 = nn.MSELoss().cuda()
criterionBCE = nn.BCELoss().cuda()

TINY = 1e-8
LossEpoch = []

epochs = 60

for i in range(epochs):
    if i ==0:
        print('Start')
    r_loss = []
    g_loss = []
    batch_index = 0
    TrainFake = True
    for batch, label in train_loader:
        if TrainFake:
            X_G = batch[:int(batch_size/2)]
            X_Real = batch[int(batch_size/2):]
            X_Real = Variable(X_Real.view([batch_size - int(batch_size/2),1,784])).cuda()
            
            X_corrupt = torch.cat((Corrupt_Gaussian(X_G[:int(batch_size/4),:,:], 0.2), Corrupt_Gaussian(X_G[int(batch_size/4):,:,:], 0.2)), dim = 0)
            X_corrupt = Variable(X_corrupt).cuda()
            X_corrupt = X_corrupt.view([-1, 1, 28 , 28])
            X_Real = X_Real.view([-1, 1 , 28 , 28])
      
            z_sample = Encoder(X_corrupt)
            X_sample = Decoder(z_sample)
            
            prediction_fake = Discriminator(X_sample)
            prediction_real = Discriminator(X_Real)
            
            #Logits
            logits0 = Variable(torch.ones(prediction_fake.size()).cuda(), requires_grad = False)
            logits1 = Variable(torch.zeros(prediction_fake.size()).cuda(), requires_grad = False)        
        
            #GAN Loss
            g_loss_GAN = criterionBCE(prediction_fake, logits1)
            
            
            X_G = Variable(X_G).cuda()
            #Reconstruction Loss                                     
            recon_loss = criterionL2(X_sample, X_G)
            total_loss = recon_loss + 0.0*g_loss_GAN
            
            Encoder_grad.zero_grad()
            Decoder_grad.zero_grad()
            total_loss.backward()
            Encoder_grad.step()
            Decoder_grad.step()
            
            r_loss.append(recon_loss.data)
            g_loss.append(g_loss_GAN.data)
            
            if batch_index == 0:
                batch_index =1
                digit_corrupt = X_corrupt[0].cpu().data.numpy().reshape([28,28])
                digit_recon = X_sample[0].cpu().data.numpy().reshape([28,28])
                plt.imshow(digit_corrupt, cmap = 'gray')
                plt.savefig('./Images/digit_corrupt',dpi = 300)
                plt.imshow(digit_recon, cmap = 'gray')
                plt.savefig('./Images/digit_recon',dpi = 300)
                digit_corrupt = X_corrupt[-1].cpu().data.numpy().reshape([28,28])
                digit_recon = X_sample[-1].cpu().data.numpy().reshape([28,28])
                plt.imshow(digit_corrupt, cmap = 'gray')
                plt.savefig('./Images/digit_corruptMask',dpi = 300)
                plt.imshow(digit_recon, cmap = 'gray')
                plt.savefig('./Images/digit_reconMask',dpi = 300)
            
            TrainFake = False
            
        else:
            pass
            X_G = batch[:int(batch_size/2)]
            X_Real = batch[int(batch_size/2):]
            X_Real = Variable(X_Real.view([batch_size - int(batch_size/2),1,784])).cuda()
            
            X_corrupt = torch.cat((Corrupt_Gaussian(X_G[:int(batch_size/4),:,:], 0.2), Corrupt_Gaussian(X_G[int(batch_size/4):,:,:],  0.2)), dim = 0)
            X_corrupt = X_corrupt.view([-1, 1, 28 , 28])
            X_Real = X_Real.view([-1, 1 , 28 , 28])
            X_corrupt = Variable(X_corrupt).cuda()
            X_G = Variable(batch[:int(batch_size/2)]).cuda()
            z_sample = Encoder(X_corrupt)
            X_sample = Decoder(z_sample)          
            
            prediction_fake = Discriminator(X_sample.detach())
            prediction_real = Discriminator(X_Real)
            
            #Logits
            logits0 = Variable(torch.ones(prediction_fake.size()).cuda(), requires_grad = False)
            logits1 = Variable(torch.zeros(prediction_fake.size()).cuda(), requires_grad = False)
            
            d_loss_GAN = (criterionBCE(prediction_real, logits1) + criterionBCE(prediction_fake, logits0))*0.5
            
            Discriminator_grad.zero_grad()
            d_loss_GAN.backward()
            Discriminator_grad.step()
            TrainFake = True
       
    

    print(i)
    print('g', np.mean(np.array(g_loss)).cpu().numpy())
    print('r', np.mean(np.array(r_loss)).cpu().numpy())
    
    
SaveName = "./NetworkSaves/Conv_" + str(z_dim) + ".pt"
torch.save(Encoder.state_dict(), SaveName)

Encoder.eval()
Decoder.eval()
    
List0x = []
List0y = []
List1x = []
List1y = []
List2x = []
List2y = []
List3x = []
List3y = []
List4x = []
List4y = []
List5x = []
List5y = []

for i in range(400):
    pair = iter(test_loader).next()
    testimg = pair[0]
    label = pair[1]
    testimg = Variable(testimg.view([1,1,28,28])).cuda()
    coor = Encoder(testimg)
    #print(corr.size())
    coor = coor[0]
    coorformat0 = coor[0][0].data.cpu().numpy()
    coorformat1 = coor[0][1].data.cpu().numpy()
    label = label.cpu().numpy()[0]
    if label == 0:
        List0x.append(coorformat0)
        List0y.append(coorformat1)
    if label == 1:
        List1x.append(coorformat0)
        List1y.append(coorformat1)
    if label == 2:
        List2x.append(coorformat0)
        List2y.append(coorformat1)
    if label == 3:
        List3x.append(coorformat0)
        List3y.append(coorformat1)
    if label == 4:
        List4x.append(coorformat0)
        List4y.append(coorformat1)
    if label == 5:
        List5x.append(coorformat0)
        List5y.append(coorformat1)

plt.clf()
plt.scatter(List0x,List0y,label='0', s=20)
plt.scatter(List1x,List1y,label='1', s=20)
plt.scatter(List2x,List2y,label='2', s=20)
plt.scatter(List3x,List3y,label='3', s=20)
plt.scatter(List4x,List4y,label='4', s=20)
plt.scatter(List5x,List5y,label='5', s=20)
plt.legend()
plt.savefig('./Images/Conv2.png',dpi = 300)
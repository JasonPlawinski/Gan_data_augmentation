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

batch_size = 100

mnist_train = datasets.MNIST('./MNIST/', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = datasets.MNIST('/MNIST/', train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=1,shuffle=True)

#Encoder
class Encoder_net(nn.Module):
    def __init__(self):
        super(Encoder_net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(X_dim, N),
            nn.Dropout(p=prob),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=prob),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=prob),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=prob),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N0),
            nn.Dropout(p=prob),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(N0, z_dim))            
           
    def forward(self, x):
        xgauss = self.layers(x)
        return xgauss
    
    # Decoder
class Decoder_net(nn.Module):
    def __init__(self):
        super(Decoder_net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(z_dim, N0),
            nn.Dropout(p=prob),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N0, N),
            nn.Dropout(p=prob),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=prob),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=prob),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=prob),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, X_dim),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        out = self.layers(x)
        return out
    
    
class Discriminator_net(nn.Module):
    def __init__(self):
        super(Discriminator_net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(X_dim, N),
            nn.Dropout(p=prob),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=prob),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=prob),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=prob),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N0),
            nn.Dropout(p=prob),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N0, 1),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        out = self.layers(x)
        return out
    
def Corrupt_Gaussian(Image, sigma):
    GN = Image + torch.normal(torch.zeros(Image.size()))*sigma
    GN[GN<0] = 0
    GN[GN>1] = 1
    return GN

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


prob = 0.2
X_dim = 784
N = 600
N0 = 150
z_dim = 2

torch.manual_seed(10)
Encoder = Encoder_net()
Decoder = Decoder_net()     # Encoder/Decoder
Discriminator = Discriminator_net()

Encoder = Encoder.cuda()
Decoder = Decoder.cuda()
Discriminator = Discriminator.cuda()
# Set learning rates
gen_lr, reg_lr = 0.0004, 0.0008
# Set optimizators
Encoder_grad = optim.Adam(Encoder.parameters(), lr=gen_lr)
Decoder_grad = optim.Adam(Decoder.parameters(), lr=gen_lr)
Discriminator_grad = optim.Adam(Discriminator.parameters(), lr=gen_lr)

criterionL2 = nn.MSELoss().cuda()

TINY = 1e-8
LossEpoch = []

for i in range(10):
    if i ==0:
        print('Start')
    loss = []
    batch_index = 0
    for batch, label in train_loader:
        X = batch.view([100,1,784])
        X_corrupt = torch.cat((Corrupt_Gaussian(X[:50,:,:], 0.2), Corrupt_Mask(X[50:,:,:], 0.007)), dim = 0)
        X = Variable(X).cuda()
        X_corrupt = Variable(X_corrupt).cuda()
        z_sample = Encoder(X_corrupt)
        X_sample = Decoder(z_sample)
        recon_loss = criterionL2(X_sample, X)
        
        if batch_index == 0:
            batch_index =1
            digit_corrupt = X_corrupt[0].cpu().data.numpy().reshape([28,28])
            digit_recon = X_sample[0].cpu().data.numpy().reshape([28,28])
            plt.imshow(digit_corrupt, cmap = 'gray')
            plt.savefig('./digit_corrupt',dpi = 300)
            plt.imshow(digit_recon, cmap = 'gray')
            plt.savefig('./digit_recon',dpi = 300)
            digit_corrupt = X_corrupt[52].cpu().data.numpy().reshape([28,28])
            digit_recon = X_sample[52].cpu().data.numpy().reshape([28,28])
            plt.imshow(digit_corrupt, cmap = 'gray')
            plt.savefig('./digit_corruptMask',dpi = 300)
            plt.imshow(digit_recon, cmap = 'gray')
            plt.savefig('./digit_reconMask',dpi = 300)
        
        Encoder_grad.zero_grad()
        Decoder_grad.zero_grad()
        recon_loss.backward()
        Encoder_grad.step()
        Decoder_grad.step()
        #loss.append(recon_loss.data)
   
    #l = np.mean(np.array(loss)).cpu().numpy()
    #print(i ,l[0])
    #LossEpoch.append(l)
    print(i)


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
    testimg = Variable(testimg.view([1,1,784])).cuda()
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
plt.savefig('./DAE20.png',dpi = 300)
np.savetxt('List', np.array(LossEpoch))
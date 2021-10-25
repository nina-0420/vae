# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 14:38:41 2021

@author: 15626
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# Ref: https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb
# Ref: https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
# Ref: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# Ref: (To plot 100 result images) https://medium.com/the-data-science-publication/how-to-plot-mnist-digits-using-matplotlib-65a2e0cc068

# To solve Intel related matplotlib/torch error.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder =torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),#64*64
        #torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),#14*14
        torch.nn.BatchNorm2d(32),
        torch.nn.LeakyReLU(0.2, inplace=True),
        
        torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),#32*32
        torch.nn.BatchNorm2d(64),
        torch.nn.LeakyReLU(0.2, inplace=True),

        torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),#32*32
        torch.nn.BatchNorm2d(128),
        torch.nn.LeakyReLU(0.2, inplace=True),        


        torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),#32*32
        torch.nn.BatchNorm2d(128),
        torch.nn.LeakyReLU(0.2, inplace=True),
        )

        self.get_mu=torch.nn.Sequential(
            torch.nn.Linear(128 * 16 * 16, 512)#32
            #torch.nn.Linear(128 * 7 * 7, 1)#32
        )
        self.get_logvar = torch.nn.Sequential(
            torch.nn.Linear(128 * 16 * 16, 512)#32
        )
        self.get_temp = torch.nn.Sequential(
            torch.nn.Linear(512, 128 * 32 * 32)#32
        )
        self.decoder = torch.nn.Sequential(
                       
        torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(inplace=True),
        
        torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(inplace=True),

        torch.nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),   
        torch.nn.Sigmoid(),
        #torch.nn.Tanh(),
        )

    def get_z(self,mu,logvar):
        eps=torch.randn(mu.size(0),mu.size(1))#64,32
        eps=torch.autograd.Variable(eps)
        if torch.cuda.is_available():
           cuda = True
           device = 'cuda'
        else:
           cuda = False
           device = 'cpu'
        eps=eps.to(device)
        z=mu+eps*torch.exp(logvar/2)
        return mu
        #return z
    
    def forward(self, x):
        out1=self.encoder(x)
        mu=self.get_mu(out1.view(out1.size(0),-1))#64,128*7*7->64,32
        out2=self.encoder(x)
        logvar=self.get_logvar(out2.view(out2.size(0),-1))

        z=self.get_z(mu,logvar)
        out3=self.get_temp(z).view(z.size(0),128,32,32)
        #out3=self.get_temp(z).view(z.size(0),128,7,7)
        #generated_imgs = z.view(z.size(0), 1, 256, 192)
        out = self.decoder(out3)
        return out, mu, logvar
    
    def kl_loss(self, mu, logvar, prior_mu=0):
        v_kl = mu.add(-prior_mu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        v_kl = v_kl.sum(dim=-1).mul_(-0.5)  # (batch, 2)
        return v_kl

    def reconstruction_loss(self, prediction, target, size_average=False):
        error = (prediction - target).view(prediction.size(0), -1)
        error = error ** 2
        error = torch.sum(error, dim=-1)
        if size_average:
            error = error.mean()
        else:
            error = error.sum()

        return error
  
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = VAE().to(device)

summary(model, (1, 128, 128))
# =============================================================================
#     def encoder(self, x):
#         dimension = x.shape[0]
#         x = x.view(dimension, 28, 28)
# 
#         hidden = self.hidden
# 
#         # Check GPU:
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         if device == 'cuda':
#             if isinstance(hidden, tuple):
#                 hidden = (hidden[0].cuda(), hidden[1].cuda())
#             else:
#                 hidden = hidden.cuda()
# 
#         x, hidden = self.lstm1(x, hidden)
#         self.hidden = hidden
# 
#         # Arrange for linear
#         x = x.reshape(100, 28*64)
#         mu = self.encFC1(x)
#         log_var = self.encFC2(x)
#         return mu, log_var
# =============================================================================

# =============================================================================
#     def sampling(self, mu, log_var):
#         std = torch.exp(log_var / 2)
#         eps = torch.randn_like(std)
#         return mu + std * eps
# =============================================================================

# =============================================================================
#     def decoder(self, x):
#         x = x.view(-1, 256, 1, 1)
#         x = self.decConv1(x)
#         x = F.relu(x)
#         x = self.conv2_bn(self.decConv2(x))
#         x = F.relu(x)
#         x = self.decConv3(x)
#         x = F.relu(x)
#         x = torch.sigmoid(self.decConv4(x))
#         return x
# =============================================================================

# =============================================================================
#     def forward(self, x):
#         # encoder -> sampling -> decoder
#         mu, log_var = self.encoder(x)
#         z = self.sampling(mu, log_var)
#         out = self.decoder(z)
#         return out, mu, log_var
# =============================================================================


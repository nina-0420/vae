# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:36:38 2021

@author: 15626
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.optim as optim
#from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from model import VAE
import argparse
import pandas as pd
from dataloader.Tagging_loader import Tagging_loader
import utils.io.image as io_func
from utils.sitk_np import np_to_sitk

# Ref: https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb
# Ref: https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
# Ref: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# Ref: (To plot 100 result images) https://medium.com/the-data-science-publication/how-to-plot-mnist-digits-using-matplotlib-65a2e0cc068

# To solve Intel related matplotlib/torch error.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(description='Tagging/VAE')
parser.add_argument('--dir_ids', type=str, default='./dataset/ukbb_roi.csv')
parser.add_argument('--percentage', type=float, default=0.80)
parser.add_argument('--batch_size', default=1, type=int)#8
parser.add_argument('--tagging_img_size', type=list, default=[128, 128, 1])#15
#parser.add_argument('--tagging_img_size', type=list, default=[192, 256, 1])
parser.add_argument('--n_cpu', default=1, type=int)
parser.add_argument('--dir_dataset', type=str, default='./dataset/')

args = parser.parse_args()
os.makedirs("vae_images", exist_ok=True)
os.makedirs("vae_test_tagging", exist_ok=True)



def loss_function(recon_x, x, mu, log_var):
    L1_loss = nn.L1Loss(reduction='sum')#
    BCE = L1_loss(recon_x, x)
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    #return BCE, BCE, KLD
    loss = BCE + 0.0000000002 * KLD
    return loss, BCE, KLD

# =============================================================================
# def Lossfunc(new_x,old_x,mu,logvar):
#     BCE=torch.nn.functional.binary_cross_entropy(new_x,old_x,size_average=False)
#     KLD=-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
#     return BCE+KLD
# =============================================================================

def train(epoch, vae, train_loader, optimizer):
    vae.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        # Check GPU:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            data = data.cuda()

        optimizer.zero_grad()

        recon_batch, mu, log_var = vae.forward(data)
        loss, bce, kld = loss_function(recon_batch, data, mu, log_var)
        #bce =  vae.reconstruction_loss(recon_batch, data, True)
        #kld = vae.kl_loss(mu, log_var).mean()
        
        #generated_imgs,mu,logvar=vae.forward(original_imgs)
        save_image(recon_batch.data[:1], "vae_images/%d-%d.png" % (epoch, batch_idx),)
        save_image(data[:1], "vae_images/%d-%d_original_imgs.png" % (epoch, batch_idx))
# =============================================================================
#         hidden = vae.hidden
#         if isinstance(hidden, tuple):
#             hidden = (hidden[0].detach(), hidden[1].detach())
#         else:
#             hidden = hidden.detach()
#         vae.hidden = hidden
# =============================================================================

        loss.backward()
        train_loss += loss.item()
        train_kld += kld
        train_bce += bce
        optimizer.step()

        #if batch_idx % 100 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    print('====> Epoch: {} Average BCE: {:.4f}'.format(epoch, train_bce / len(train_loader.dataset)))
    print('====> Epoch: {} Average KLD: {:.4f}'.format(epoch, train_kld / len(train_loader.dataset)))

    loss = train_loss / len(train_loader.dataset)
    return_kld = train_kld / len(train_loader.dataset)
    return_bce = train_bce / len(train_loader.dataset)

    return loss, return_kld.__float__(), return_bce.__float__()


def test(epoch, vae, test_loader):
    vae.eval()
    test_loss = 0

    # Check GPU:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for test_batch_index, (data, _) in enumerate(test_loader):
        #for data, _ in test_loader:
            if device == 'cuda':
                data = data.cuda()
            recon, mu, log_var = vae(data)

            # sum up batch loss
            loss, _, _ = loss_function(recon, data, mu, log_var)
            ##bce =  vae.reconstruction_loss(recon, data, True)
            ##kld = vae.kl_loss(mu, log_var).mean()
            ##loss = bce + 0.00000000002 * kld
            save_image(recon[:1], "vae_test_tagging/%d-%d.png" % (epoch, test_batch_index),)
            save_image(data[:1], "vae_test_tagging/%d-%d_original_imgs.png" % (epoch, test_batch_index))
            test_loss += loss.item()

        test_loss /= len(test_loader)
        print('====> Test set loss: {:.4f}'.format(test_loss))
    
    
    return test_loss

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    print('\nLoading IDs file\n')
    IDs = pd.read_csv(args.dir_ids, sep=',')
    # Dividing the number of images for training and test.
    IDs_copy = IDs.copy()
    train_set = IDs_copy.sample(frac = args.percentage, random_state=0)
    test_set = IDs_copy.drop(train_set.index)
    train_loader = Tagging_loader(batch_size = args.batch_size,
                         tagging_img_size= args.tagging_img_size,
                         num_workers = args.n_cpu,
                         #sax_img_size = args.sax_img_size,
            			 shuffle = True,
            			 dir_imgs = args.dir_dataset,
                         args = args,
                         ids_set = train_set
            			  )

    test_loader = Tagging_loader(batch_size = args.batch_size,
                            tagging_img_size= args.tagging_img_size,
                        	num_workers = args.n_cpu,
                            #sax_img_size = args.sax_img_size,
            			    shuffle = True,
            			    dir_imgs = args.dir_dataset,
                            args = args,
                            ids_set = test_set
            			     )
    
# =============================================================================
#     bs = 100
#     # MNIST Dataset
#     train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
#     test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)
# 
#     # Data Loader (Input Pipeline)
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)
# =============================================================================
    P_learning_rate=0.00001
    vae = VAE()
    if torch.cuda.is_available():
        vae.cuda()
        
    print('The structure of our model is shown below: \n')
    print(vae)

    optimizer = optim.Adam(vae.parameters(), lr=P_learning_rate)
    #optimizer = optim.Adam(vae.parameters())

    # Training:
    train_loss_to_plot = []
    kl_div_to_plot = []
    bce_to_plot = []
    test_loss_to_plot = []
    for epoch in range(1, 30):
        train_loss, kld, bce = train(epoch, vae, train_loader, optimizer)
        train_loss_to_plot.append(train_loss)
        kl_div_to_plot.append(kld)
        bce_to_plot.append(bce)

        test_loss = test(epoch, vae, test_loader)
        test_loss_to_plot.append(test_loss)
    # Saving the trained model:
    PATH = './model_new.pth'
    torch.save(vae.state_dict(), PATH)

    # show loss curve
    plt.plot(train_loss_to_plot)
    plt.show()
    plt.savefig('loss_train.png')

    # show KL divergence curve
    plt.plot(kl_div_to_plot)
    plt.show()
    plt.savefig('klloss_train.png')

    # show BCE curve
    plt.plot(bce_to_plot)
    plt.show()
    plt.savefig('bceloss_train.png')
    
    # show test loss curve
    plt.plot(test_loss_to_plot)
    plt.show()
    plt.savefig('loss_test.png')

if __name__ == '__main__':
    main()
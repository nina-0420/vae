# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 16:31:21 2021

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
# import EarlyStopping
from pytorchtools import EarlyStopping
from model_cine_allframe import VAE
import argparse
import pandas as pd
from dataloader.Cine_loader import Cine_loader
import utils.io.image as io_func
from utils.sitk_np import np_to_sitk
import numpy as np
import shutil
import sys
sys.path.append('/usr/not-backed-up/scnc/MNIST-VAE-main/dataloader/Cine_loader.py') 


# Ref: https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb
# Ref: https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
# Ref: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# Ref: (To plot 100 result images) https://medium.com/the-data-science-publication/how-to-plot-mnist-digits-using-matplotlib-65a2e0cc068

# To solve Intel related matplotlib/torch error.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(description='Cine/VAE')
parser.add_argument('--dir_ids', type=str, default='./dataset/ukbb_roi_10.csv')
parser.add_argument('--percentage', type=float, default=0.80)
parser.add_argument('--percentage1', type=float, default=0.30)
parser.add_argument('--batch_size', default=4, type=int)#8
parser.add_argument('--sax_img_size', type=list, default=[128, 128, 12])#15
#parser.add_argument('--tagging_img_size', type=list, default=[192, 256, 1])
parser.add_argument('--n_cpu', default=0, type=int)
parser.add_argument('--dir_dataset', type=str, default='./dataset/')

args = parser.parse_args()
os.makedirs("vae_images_cine", exist_ok=True)
os.makedirs("vae_test_cine", exist_ok=True)
#os.makedirs("vae_images_cine")
#os.makedirs("vae_test_cine")



def loss_function(recon_x, x, mu, log_var):
    L1_loss = nn.L1Loss(reduction='sum')#
    BCE = L1_loss(recon_x, x)
    L2_loss = nn.MSELoss()
    MSE = L2_loss (recon_x, x)
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    #return BCE + KLD, BCE, KLD
    #loss = BCE+ 0.000000000002 * KLD
    loss = BCE
    return loss, BCE, KLD

# =============================================================================
# def Lossfunc(new_x,old_x,mu,logvar):
#     BCE=torch.nn.functional.binary_cross_entropy(new_x,old_x,size_average=False)
#     KLD=-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
#     return BCE+KLD
# =============================================================================
def train_model(vae, batch_size, patience, n_epochs, train_loader, valid_loader, optimizer):
    
    # to track the training loss as the model trains
    train_losses = []
    train_bce = []
    train_kld = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        vae.train() # prep model for training
        
        for batch_idx, (data, _, mean, std, maxx) in enumerate(train_loader):
            # Check GPU:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
               data = data.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            recon_batch, mu, log_var = vae.forward(data)
            # calculate the loss
            loss, bce, kld = loss_function(recon_batch, data, mu, log_var)          
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())
            train_kld.append(kld.item())
            train_bce.append(bce.item())
            data_denor_original = data[:1].numpy()
            data_denor_original = data_denor_original[0:,]

            std = std[:1].numpy()
            mean = mean[:1].numpy()
            maxx = maxx[:1].numpy()
            
            data_denor = recon_batch.data[:1].numpy()
            data_denor = data_denor[0:,]
            data_denor = data_denor * maxx
            data_denor = data_denor * std
            data_denor = data_denor + mean
            #data_denor = data_denor * 255
            data_denor = torch.from_numpy(data_denor)
           
            data_denor_original = data_denor_original * maxx
            data_denor_original = data_denor_original * std
            data_denor_original = data_denor_original + mean
            data_denor_original = torch.from_numpy(data_denor_original)
            #data_denor_original = data_denor_original * 255
            #save_image(data_denor, "vae_images_cine/%d-%d_denor.png" % (epoch, batch_idx),)
            #save_image(data_denor_original, "vae_images_cine/%d-%d_denor_ori.png" % (epoch, batch_idx),)
            save_image(recon_batch.data[:1], "vae_images_cine/%d-%d.png" % (epoch, batch_idx),)
            save_image(data[:1], "vae_images_cine/%d-%d_original_imgs.png" % (epoch, batch_idx))
            #io_func.write(np_to_sitk(data[:1].cpu().detach().numpy()[0]), "vae_images_cine/%d-%d_original_imgs.nii.gz" % (epoch, batch_idx))
            #io_func.write(np_to_sitk(recon_batch.data[:1].cpu().detach().numpy()[0]), "vae_images_cine/%d-%d.nii.gz" % (epoch, batch_idx))
            #io_func.write(np_to_sitk(data_denor.cpu().detach().numpy()[0]), "vae_images_cine/%d-%d_denor.nii.gz" % (epoch, batch_idx))
            #io_func.write(np_to_sitk(data_denor_original.cpu().detach().numpy()[0]), "vae_images_cine/%d-%d_denor_ori.nii.gz" % (epoch, batch_idx))
        ######################    
        # validate the model #
        ######################
        vae.eval() # prep model for evaluation
        for batch_idx, (data, _, mean, std, maxx) in enumerate(valid_loader):

            # forward pass: compute predicted outputs by passing inputs to the model
            recon_batch, mu, log_var = vae.forward(data)
            # calculate the loss
            loss, bce, kld = loss_function(recon_batch, data, mu, log_var)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, vae)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    vae.load_state_dict(torch.load('checkpoint.pt'))

    return  vae, avg_train_losses, avg_valid_losses

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
        #generated_imgs,mu,logvar=vae.forward(original_imgs)
        data_denor = recon_batch.data[:2].numpy()
        data_denor = data_denor[0:,]
        #data_denor = data_denor * np.max(data_denor)
        data_denor = data_denor * np.std(data_denor)
        data_denor = data_denor + np.mean(data_denor)
        data_denor = torch.from_numpy(data_denor)
        
        data_denor_original = data[:2].numpy()
        data_denor_original = data_denor_original[0:,]
        #data_denor_original = data_denor_original * np.max(data_denor_original)
        data_denor_original = data_denor_original * np.std(data_denor_original)
        data_denor_original = data_denor_original + np.mean(data_denor_original)
        data_denor_original = torch.from_numpy(data_denor_original)
        save_image(data_denor_original, "vae_images_cine/%d-%d_denor_ori.png" % (epoch, batch_idx),)
        save_image(data_denor, "vae_images_cine/%d-%d_denor.png" % (epoch, batch_idx),)
        save_image(recon_batch.data[:2], "vae_images_cine/%d-%d.png" % (epoch, batch_idx),)
        save_image(data[:2], "vae_images_cine/%d-%d_original_imgs.png" % (epoch, batch_idx))
# =============================================================================
#         hidden = vae.hidden
#         if isinstance(hidden, tuple):
#             hidden = (hidden[0].detach(), hidden[1].detach())
#         else:
#             hidden = hidden.detach()
#         vae.hidden = hidden
# =============================================================================
# backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        train_loss += loss.item()
        train_kld += kld
        train_bce += bce
        # perform a single optimization step (parameter update)
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
            test_loss += loss.item()
            save_image(recon[:1], "vae_test_cine/%d-%d.png" % (epoch, test_batch_index),)
            save_image(data[:1], "vae_test_cine/%d-%d_original_imgs.png" % (epoch, test_batch_index))

        test_loss /= len(test_loader)
        print('====> Test set loss: {:.4f}'.format(test_loss))
    
    return test_loss

def main():
    
    #manual seed
    torch.manual_seed(128)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    print('\nLoading IDs file\n')
    IDs = pd.read_csv(args.dir_ids, sep=',')
    # Dividing the number of images for training and test.
    IDs_copy = IDs.copy()
    train_set = IDs_copy.sample(frac = args.percentage, random_state=0)
    test_set = IDs_copy.drop(train_set.index)
    val_set = train_set.sample(frac = args.percentage1, random_state=0)
    train_set = train_set.drop(val_set.index)
    print('train:', len(train_set), 'test:', len(test_set))

    print('train:', len(train_set), 'validation:', len(val_set))
    train_loader = Cine_loader(batch_size = args.batch_size,
                         #sax_img_size= args.sax_img_size,
                         num_workers = args.n_cpu,
                         sax_img_size = args.sax_img_size,
            			 shuffle = True,
            			 dir_imgs = args.dir_dataset,
                         args = args,
                         ids_set = train_set
            			  )
    
    val_loader = Cine_loader(batch_size = args.batch_size,
                         #sax_img_size= args.sax_img_size,
                         num_workers = args.n_cpu,
                         sax_img_size = args.sax_img_size,
            			 shuffle = True,
            			 dir_imgs = args.dir_dataset,
                         args = args,
                         ids_set = val_set
            			  )

    test_loader = Cine_loader(batch_size = args.batch_size,
                            #tagging_img_size= args.tagging_img_size,
                        	num_workers = args.n_cpu,
                            sax_img_size = args.sax_img_size,
            			    shuffle = False,
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
    
    batch_size = 256
    n_epochs = 200
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 20
    P_learning_rate=0.00001
    vae = VAE()
    if torch.cuda.is_available():
        vae.cuda()
        
    print('The structure of our model is shown below: \n')
    print(vae)
    
    optimizer = optim.Adam(vae.parameters(), lr=P_learning_rate)
    #optimizer = optim.Adam(vae.parameters())
    # Training:
    vae, train_loss, valid_loss = train_model(vae, batch_size, patience, n_epochs, train_loader, val_loader, optimizer)
    
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
    
    # find position of lowest validation loss
# =============================================================================
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    plt.xlabel('epochs')
    plt.ylabel('loss')
#    plt.ylim(0, 0.5) # consistent scale
#    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')
    
    #Saving the trained model:
    PATH = './model_new_cine.pth'
    torch.save(vae.state_dict(), PATH)
    
# =============================================================================
# =============================================================================
#     train_loss_to_plot = []
#     kl_div_to_plot = []
#     bce_to_plot = []
#     test_loss_to_plot = []
#     for epoch in range(1, 300):
#         train_loss, kld, bce = train(epoch, vae, train_loader, optimizer)
#         train_loss_to_plot.append(train_loss)
#         kl_div_to_plot.append(kld)
#         bce_to_plot.append(bce)
# 
#         test_loss = test(epoch, vae, test_loader, patience)
#         test_loss_to_plot.append(test_loss)
# 
#     # Saving the trained model:
#     PATH = './model_new_cine.pth'
#     torch.save(vae.state_dict(), PATH)
# 
#     # show loss curve
#     plt.plot(train_loss_to_plot)
#     plt.show()
# 
#     # show KL divergence curve
#     plt.plot(kl_div_to_plot)
#     plt.show()
# 
#     # show BCE curve
#     plt.plot(bce_to_plot)
#     plt.show()
#     
#     # show test loss curve
#     plt.plot(test_loss_to_plot)
#     plt.show()
# =============================================================================

if __name__ == '__main__':
    main()
    
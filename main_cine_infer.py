# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 22:43:28 2021

@author: scnc
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
from model_cine import VAE
import argparse
import pandas as pd
from dataloader.Cine_loader import Cine_loader
import utils.io.image as io_func
from utils.sitk_np import np_to_sitk
import numpy as np
import shutil
import sys
sys.path.append('/usr/not-backed-up/scnc/MNIST-VAE-main/dataloader/Cine_loader.py')

# To solve Intel related matplotlib/torch error.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(description='Cine/VAE')
parser.add_argument('--dir_ids', type=str, default='./dataset/ukbb_roi_10.csv')
parser.add_argument('--percentage', type=float, default=0.80)
parser.add_argument('--percentage1', type=float, default=0.30)
parser.add_argument('--batch_size', default=4, type=int)#8
parser.add_argument('--sax_img_size', type=list, default=[128, 128, 1])#15
#parser.add_argument('--tagging_img_size', type=list, default=[192, 256, 1])
parser.add_argument('--n_cpu', default=0, type=int)
parser.add_argument('--dir_dataset', type=str, default='./dataset/')

args = parser.parse_args()
os.makedirs("vae_test_cine", exist_ok=True)

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

if __name__ == '__main__':

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
    
    vae = VAE()
    if torch.cuda.is_available():
        vae.cuda()
        
    print('The structure of our model is shown below: \n')
    print(vae)
    
    PATH = './model_new_cine.pth'
    model_dict=vae.load_state_dict(torch.load(PATH))
    
    vae.eval()
    test_loss = 0
    # Check GPU:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for test_batch_index, (data, _, mean, std, maxx) in enumerate(test_loader):
        #for data, _ in test_loader:
            if device == 'cuda':
                data = data.cuda()
            recon, mu, log_var = vae(data)

            # sum up batch loss
            loss, _, _ = loss_function(recon, data, mu, log_var)
            test_loss += loss.item()
            save_image(recon[:1], "vae_test_cine/%d.png" % ( test_batch_index),)
            save_image(data[:1], "vae_test_cine/%d_original_imgs.png" % ( test_batch_index))

        test_loss /= len(test_loader)
        print('====> Test set loss: {:.4f}'.format(test_loss))
    
    
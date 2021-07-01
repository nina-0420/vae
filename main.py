# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 15:44:05 2021

@author: scnc
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 23:18:58 2021

@author: scnc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import numpy as np
import math 
import argparse
import pandas as pd
import datetime
import shutil
import utils.io.image as io_func
from utils.sitk_np import np_to_sitk
from dataloader.Tagging_loader import Tagging_loader
#from vae import plot, diagnostics
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


parser = argparse.ArgumentParser(description='Cine/VAE')
parser.add_argument('--dir_ids', type=str, default='./data/ukbb_roi.csv')
parser.add_argument('--batch_size', default=1, type=int)#8
parser.add_argument('--tagging_img_size', type=list, default=[128, 128, 1])#15
#parser.add_argument('--tagging_img_size', type=list, default=[192, 256, 1])#15
parser.add_argument('--sax_img_size', type=list, default=[128, 128, 1])
parser.add_argument('--n_cpu', default=0, type=int)
parser.add_argument('--dir_dataset', type=str, default='./dataset/')
parser.add_argument('--percentage', type=float, default=0.80)
parser.add_argument('--dir_results', type=str, default='./results/')
parser.add_argument('--test_mode', type=bool, default=False)
parser.add_argument('--dir_test_ids', type=str, default='2021-05-24_21-20-46/')
parser.add_argument('--save_model', default=10, type=int) # save the model every x epochs
parser.add_argument('--test_every', type=int, default=1, metavar='N', help='test after every epochs')
parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to latest checkpoint(default: None)')
parser.add_argument('--result_dir', type=str, default='./VAEResult', metavar='DIR', help='output directory')
parser.add_argument('--z_dim', type=int, default=20, metavar='N', help='the dim of latent variable z(default: 20)')
parser.add_argument('--save_dir', type=str, default='./checkPoint', metavar='N', help='model saving directory')
parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train(default: 200)')
args = parser.parse_args()

os.makedirs("vae_test_tagging_cov", exist_ok=True)
os.makedirs("vae_train_tagging_cov", exist_ok=True)



#P_batch_size=128
P_learning_rate=0.00003
#P_save_img_interval=200

# =============================================================================
# if torch.cuda.is_available():
#     cuda = True
#     device = 'cuda'
# else:
#     cuda = False
#     device = 'cpu'
# =============================================================================
    
device = torch.device("cpu")

def save_checkpoint(state, is_best, outdir):
    """
    每训练一定的epochs后， 判断损失函数是否是目前最优的，并保存模型的参数
    :param state: 需要保存的参数，数据类型为dict
    :param is_best: 说明是否为目前最优的
    :param outdir: 保存文件夹
    :return:
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    checkpoint_file = os.path.join(outdir, 'checkpoint.pth')  # join函数创建子文件夹，也就是把第二个参数对应的文件保存在'outdir'里
    best_file = os.path.join(outdir, 'model_best.pth')
    torch.save(state, checkpoint_file)  # 把state保存在checkpoint_file文件夹中
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)

def test(model, optimizer, test_loader, epoch, best_test_loss):
    test_avg_loss = 0.0
    
    with torch.no_grad():  # 这一部分不计算梯度，也就是不放入计算图中去
        '''测试测试集中的数据'''
        # 计算所有batch的损失函数的和
        test_loss_batch = []
        for test_batch_index, (test_x, _) in enumerate(test_loader):
            test_x = test_x.to(device)
            #original_imgs_test = torch.flatten(test_x, start_dim=1)
            # 前向传播
            original_imgs_test=torch.autograd.Variable(test_x).to(device)
            generated_imgs_test,test_mu, test_log_var=model.forward(original_imgs)
            test_loss, test_BCE, test_KLD = loss_function(generated_imgs_test, original_imgs_test, test_mu, test_log_var)
            #vae_loss, gene_loss, KLD = Lossfunc(generated_imgs_unformat, original_imgs, mu, logvar)
            # 前向传播
            #test_x_hat, test_mu, test_log_var = model(test_x)
            # 损害函数值
            #test_loss, test_BCE, test_KLD = loss_function(generated_imgs_unformat_test, original_imgs_test, test_mu, test_mu)
            #test_loss.backward()
            optimizer.step()
            save_image(generated_imgs_test[:1], "vae_test_tagging_cov/%d-%d.png" % (epoch, test_batch_index),)
            #save_image(random_res[:1], "vae_test_tagging_cov/%d-%d.png" % (epoch, test_batch_index))
            save_image(test_x[:1], "vae_test_tagging_cov/%d-%d_original_imgs.png" % (epoch, test_batch_index))
            #save_image(random_res, './%s/random_sampled-%d.png' % (args.result_dir, epoch + 1)) 
            test_loss_batch.append(test_loss.item())  # loss是Tensor类型
            test_avg_loss += test_loss
        #test_avg_loss += test_loss
            
            # 对和求平均，得到每一张图片的平均损失
        test_avg_loss /= len(test_loader)
        
        # 把这一个epoch的每一个样本的平均损失存起来
        test_losses.append(test_avg_loss.item())
        test_losses.append(np.sum(test_loss_batch) / len(test_loader))# len(mnist_train.dataset)为样本个数
            #test_losses.append(np.sum(test_loss) / len(test_loader))# len(mnist_train.dataset)为样本个数
        print('====> Test set loss: {:.4f}'.format(test_avg_loss))
          

        '''测试随机生成的隐变量'''
            # 随机从隐变量的分布中取隐变量
        z = torch.randn(args.batch_size, args.z_dim).to(device)  # 每一行是一个隐变量，总共有batch_size行
            # 对隐变量重构
            #random_res = model.decode(z).view(-1, 1, 128, 128)
            # 保存重构结果
        

        '''保存目前训练好的模型'''
            # 保存模型
        is_best = test_avg_loss < best_test_loss
        best_test_loss = min(test_avg_loss, best_test_loss)
        save_checkpoint({
               'epoch': epoch,  # 迭代次数
               'best_test_loss': best_test_loss,  # 目前最佳的损失函数值
               'state_dict': model.state_dict(),  # 当前训练过的模型的参数
               'optimizer': optimizer.state_dict(),
            }, is_best, args.save_dir)

        return best_test_loss , test_loss  


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder =torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),#64*64
        #torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),#14*14
        torch.nn.BatchNorm2d(64),
        torch.nn.LeakyReLU(0.2, inplace=True),

        torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),#32*32
        torch.nn.BatchNorm2d(128),
        torch.nn.LeakyReLU(0.2, inplace=True),

        torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),#32*32
        torch.nn.BatchNorm2d(128),
        torch.nn.LeakyReLU(0.2, inplace=True),
        )

        self.get_mu=torch.nn.Sequential(
            torch.nn.Linear(128 * 32 * 32, 32)#32
            #torch.nn.Linear(128 * 7 * 7, 1)#32
        )
        self.get_logvar = torch.nn.Sequential(
            torch.nn.Linear(128 * 32 * 32, 32)#32
        )
        self.get_temp = torch.nn.Sequential(
            torch.nn.Linear(32, 128 * 32 * 32)#32
        )
        self.decoder = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(inplace=True),

        torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
        torch.nn.Sigmoid()
        )
         

    def get_z(self,mu,logvar):
        eps=torch.randn(mu.size(0),mu.size(1))#64,32
        eps=torch.autograd.Variable(eps)
        z=mu+eps*torch.exp(logvar/2)
        return z
    

    def forward(self, x):
        #mu = torch.randn(1,requires_grad= True).to(device = 'cuda')
        #logvar = torch.randn(1,requires_grad= True).to(device = 'cuda')
        out1=self.encoder(x)
        mu=self.get_mu(out1.view(out1.size(0),-1))#64,128*7*7->64,32
        out2=self.encoder(x)
        logvar=self.get_logvar(out2.view(out2.size(0),-1))
        
        z=self.get_z(mu,logvar)
        ###z=torch.zeros(1, 32)
        out3=self.get_temp(z).view(z.size(0),128,32,32)
        #out3=self.get_temp(z).view(z.size(0),128,7,7)
        #generated_imgs = z.view(z.size(0), 1, 256, 192)

        return self.decoder(out3),mu,logvar


# =============================================================================
# def loss_function(generated, origin, mu, log_var):
#     """
#     Calculate the loss. Note that the loss includes two parts.
#     :param x_hat:
#     :param x:
#     :param mu:
#     :param log_var:
#     :return: total loss, BCE and KLD of our model
#     """
#     # 1. the reconstruction loss.
#     # We regard the MNIST as binary classification
#     BCE = F.binary_cross_entropy(generated, origin, reduction='sum')
# 
#     # 2. KL-divergence
#     # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
#     # here we assume that \Sigma is a diagonal matrix, so as to simplify the computation
#     KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)
# 
#     # 3. total loss
#     loss = BCE + KLD
#     return loss, BCE, KLD
# =============================================================================
def loss_function(generated, origin, mu, log_var):
    """
    Calculate the loss. Note that the loss includes two parts.
    :param x_hat:
    :param x:
    :param mu:
    :param log_var:
    :return: total loss, BCE and KLD of our model
    """
    # 1. the reconstruction loss.
    # We regard the MNIST as binary classification
    L1_loss = nn.L1Loss(reduction='sum')#
    gene_loss = L1_loss(generated, origin)
    #BCE = F.binary_cross_entropy(generated, origin, reduction='sum')

    # 2. KL-divergence
    # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
    # here we assume that \Sigma is a diagonal matrix, so as to simplify the computation
    KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)

    # 3. total loss
    #loss = gene_loss + KLD
    loss = gene_loss 
    #loss = BCE + KLD
    return loss, gene_loss, KLD

  
print('\nLoading IDs file\n')
IDs = pd.read_csv(args.dir_ids, sep=',')

# Dividing the number of images for training and test.
IDs_copy = IDs.copy()
train_set = IDs_copy.sample(frac = args.percentage, random_state=0)
test_set = IDs_copy.drop(train_set.index)
test_set.to_csv(args.dir_results + 'test_set.csv', index=False)
train_set.to_csv(args.dir_results + 'train_set.csv', index=False)

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
            			    shuffle = False,
            			    dir_imgs = args.dir_dataset,
                            args = args,
                            ids_set = test_set
            			     )

#dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=P_batch_size, shuffle=True)
model = VAE().to(device)
vae = VAE()
# =============================================================================
# if cuda:
#        vae.cuda() 
# =============================================================================
print('The structure of our model is shown below: \n')
print(model)
# use Adaptive moment estimation to replace normal gradient descent to get better results
optimizer = torch.optim.Adam(vae.parameters(), lr=P_learning_rate)

# Step 3: optionally resume(恢复) from a checkpoint
start_epoch = 0
best_test_loss = np.finfo('f').max
if args.resume:
        if os.path.isfile(args.resume):
            # 载入已经训练过的模型参数与结果
            print('=> loading checkpoint %s' % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s' % args.resume)
        else:
            
            
            print('=> no checkpoint found at %s' % args.resume)

if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

loss_epoch = []
test_avg_loss = []
test_losses = []

for epoch in range(start_epoch, args.epochs):
        loss_batch = []
        for i, (data) in enumerate(train_loader):
            
            tagging = data[0]
            tagging = tagging.to(device)
            '''
        if cuda:
            imgs, labels = imgs.to('cuda'), labels.to('cuda')  
        else:
            imgs, labels = imgs.to('cpu'), labels.to('cpu')
        '''
            
            optimizer.zero_grad()# 梯度清零，否则上一步的梯度仍会存在
            original_imgs=torch.autograd.Variable(tagging).to(device)
            generated_imgs,mu,logvar=vae.forward(original_imgs)  
            #vae_loss, gene_loss, KLD = Lossfunc(generated_imgs_unformat, original_imgs, mu, logvar)
            vae_loss, gene_loss, KLD = loss_function(generated_imgs, original_imgs, mu, logvar)
            #vae_loss, gene_loss, KLD = loss_function(generated_imgs_unformat, original_imgs, mu, logvar)  # 计算损失值，即目标函数 
            loss_batch.append(vae_loss.item())# loss是Tensor类型
            vae_loss.backward()# 后向传播计算梯度，这些梯度会保存在model.parameters里面
            optimizer.step()## 更新梯度，这一步与上一步主要是根据model.parameters联系起来了
           

            print("epoch %d - batch %d " % (epoch, i))
            print('Epoch[{}/{}], Batch [{}/{}] : Total-loss = {:.4f}, BCE-Loss = {:.4f}, KLD-loss = {:.4f}'
                      .format(epoch + 1, args.epochs, i + 1, len(train_loader) // args.batch_size,
                              vae_loss.item() / args.batch_size, gene_loss.item() / args.batch_size,
                              KLD.item() / args.batch_size))

            save_image(generated_imgs.data[:1], "vae_train_tagging_cov/%d-%d.png" % (epoch + 1, i + 1),)
            #save_image(generated_imgs.data[:1], "vae_train_tagging_cov/%d-%d.png" % (epoch, i),)
            save_image(tagging[:1], "vae_train_tagging_cov/%d-%d_original_imgs.png" % (epoch + 1, i + 1))
        
       # 把这一个epoch的每一个样本的平均损失存起来
        loss_epoch.append(np.sum(loss_batch) / len(train_loader))# len(mnist_train.dataset)为样本个数
        
              
        # 测试模型
        if (epoch + 1) % args.test_every == 0:
            best_test_loss, test_loss = test(model, optimizer, test_loader, epoch, best_test_loss)
               
            test_avg_loss.append(test_loss.item())
plt.figure()  
plt.plot(loss_epoch)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss_train.png')
plt.show()

plt.figure()  
plt.plot(test_avg_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss_test.png')
plt.show()        
# =============================================================================
# plt.plot(loss_epoch)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()
# plt.savefig('loss_train.png')
# =============================================================================

#在同一副图上显示
# =============================================================================
# plt.title("Training and Validation Loss")
# plt.plot(test_losses,label="val")
# plt.plot(loss_epoch,label="train")
# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.savefig('loss.png')
# plt.show()
# =============================================================================

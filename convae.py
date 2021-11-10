import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
import argparse

class encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.convolution = nn.Sequential(
                                 nn.Conv2d(1,4,3,padding=1),
                                 nn.MaxPool2d(2,stride=2),
                                 nn.Tanh(),                   # 32 x 32
                                 nn.Conv2d(4,8,3,padding=1),
                                 nn.MaxPool2d(2,stride=2),
                                 nn.Tanh(),                   # 16 X 16 
                                 nn.Conv2d(8,16,3,padding=1),
                                 nn.MaxPool2d(2,stride=2),
                                 nn.Tanh(),                   # 8 X 8 
                                 nn.Conv2d(16,32,3,padding=1),
                                 nn.MaxPool2d(2,stride=2),
                                 nn.Tanh(),                   # 4 X 4
                                 nn.Conv2d(32,64,3,padding=1),
                                 nn.MaxPool2d(2,stride=2),
                                 nn.Tanh(),                   # 2 X 2
    )
    self.linear = nn.Sequential(
                                 nn.Linear(256,128),
                                 nn.Tanh(),
    )
  def forward(self,x):
    bs = x.shape[0]
    x = self.convolution(x.unsqueeze(1))
    x = self.linear(x.reshape(bs,-1))
    return x
  
class decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.convolution = nn.Sequential(
                                 nn.ConvTranspose2d(64,32,3,padding=1,stride=2,output_padding=1),
                                 nn.Tanh(),                   # 4 x 4
                                 nn.ConvTranspose2d(32,16,3,padding=1,stride=2,output_padding=1),
                                 nn.Tanh(),                   # 8 x 8
                                 nn.ConvTranspose2d(16,8,3,padding=1,stride=2,output_padding=1),
                                 nn.Tanh(),                   # 16 x 16
                                 nn.ConvTranspose2d(8,4,3,padding=1,stride=2,output_padding=1),
                                 nn.Tanh(),                   # 32 x 32
                                 nn.ConvTranspose2d(4,1,3,padding=1,stride=2,output_padding=1),
                                 nn.Tanh(),                   # 64 x 64
                                 nn.ConvTranspose2d(1,1,3,padding=1),
    )
    self.linear = nn.Sequential(
                                 nn.Linear(128,256),
                                 nn.Tanh(),
    )
  def forward(self,x):
    bs = x.shape[0]
    n = x.shape[1]
    x = self.linear(x).reshape(bs,64,2,2)
    x = self.convolution(x)
    return x.squeeze(1)

#function to initialize model weights with kaiming method:
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data,0,mode='fan_out')
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data,0,mode='fan_out')

#Build Autoencoder

class Autoencoder(pl.LightningModule):
  def __init__(self, lr = 1e-3):
    super().__init__()
    self.lr = lr
    self.encoder = encoder()
    self.encoder.apply(weights_init)
    self.decoder = decoder()
    self.decoder.apply(weights_init)  
  def forward(self,x):
    return self.decoder(self.encoder(x))
  
  def loss(self,y,x):
    return F.mse_loss(x,y)
  
  def training_step(self,batch,batch_idx):
    orig = batch
    recon = self(orig)
    train_loss = self.loss(orig,recon)
    self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return train_loss

  def validation_step(self, batch, batch_idx):
    orig = batch
    recon = self(orig)
    val_loss = torch.sqrt(F.mse_loss(orig,recon)/torch.mean(orig**2))
    self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return val_loss
  
  def configure_optimizers(self):
      opt =  torch.optim.Adam(self.parameters(), lr=self.lr)
      return  {
        'optimizer': opt,
        'lr_scheduler': {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=20, threshold=1e-2, min_lr=1e-5),
            'monitor': 'val_loss',
        }
    }
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.utils.tensorboard
import argparse
import h5py
import scipy

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

class mydataset(torch.utils.data.Dataset):
  def __init__(self,data):
    self.data = data
    nx = self.data.shape[1]
    ny = self.data.shape[2]
    self.data = self.data.permute(3,0,1,2).reshape(-1,nx,ny)
  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self,idx):
    return self.data[idx]
      
class ns2d_dm(pl.LightningDataModule):
  
  def __init__(self,path,batch_size):
    super().__init__()
    self.path = path
    self.batch_size = batch_size
    self.reader = MatReader(path, to_torch=True)
    
  def prepare_data(self):
    return None
  def setup(self,stage=None):
    data = self.reader.read_field('u')[:,:,:,1:]
    self.train_data, self.val_data = [mydataset(data[:data.shape[0]*9//10]),  mydataset(data[data.shape[0]*9//10:])]

  def train_dataloader(self):
    return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=2,shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=2)

  def test_dataloader(self):
    return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=2)

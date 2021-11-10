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
from convae import Autoencoder
import argparse

## Utilities
@torch.jit.script
def fuse_mul_add_mul(f, cell_states, i, g):
    return f * cell_states + i * g

def chkpt_blk(cc_i, cc_f, cc_o, cc_g, cell_states):
    i = torch.sigmoid(cc_i)
    f = torch.sigmoid(cc_f)
    o = torch.sigmoid(cc_o)
    g = torch.tanh(cc_g)
    
    cell_states = fuse_mul_add_mul(f, cell_states, i, g)
    outputs = o * torch.tanh(cell_states)

    return outputs, cell_states

## Standard Convolutional-LSTM Module
class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size = 5, bias = True):
        """
        Construction of convolutional-LSTM cell.
        
        Arguments:
        ----------
        (Hyper-parameters of input/output interfaces)
        input_channels: int
            Number of channels of the input tensor.
        hidden_channels: int
            Number of channels of the hidden/cell states.

        (Hyper-parameters of the convolutional opeations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 3
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True

        """
        super(ConvLSTMCell, self).__init__()

        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels

        kernel_size = utils._triple(kernel_size)
        padding     = kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2

        self.conv = nn.Conv1d(
            in_channels  = input_channels + hidden_channels, 
            out_channels = 4 * hidden_channels,
            kernel_size = kernel_size, padding = padding, bias = bias)

        # Note: hidden/cell states are not intialized in construction
        self.hidden_states, self.cell_state = None, None

    def initialize(self, inputs):
        """
        Initialization of convolutional-LSTM cell.
        
        Arguments: 
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, input_channels, input_height, input_width]
            Input tensor of convolutional-LSTM cell.

        """
        device = inputs.device # "cpu" or "cuda"
        batch_size, _, height = inputs.size()

        # initialize both hidden and cell states to all zeros
        self.hidden_states = torch.zeros(batch_size, 
            self.hidden_channels, height, device = device)
        self.cell_states   = torch.zeros(batch_size, 
            self.hidden_channels, height, device = device)

    def forward(self, inputs, first_step = False, checkpointing = False):
        """
        Computation of convolutional-LSTM cell.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, input_channels, height, width] 
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the tensor is the first step in the input sequence. 
            Note: If so, both hidden and cell states are intialized to zeros tensors.
            default: False

        checkpointing: bool
            Whether to use the checkpointing technique to reduce memory expense.
            default: True
        
        Returns:
        --------
        hidden_states: another 4-th order tensor of size 
            [batch_size, hidden_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.

        """
        if first_step: self.initialize(inputs)

        concat_conv = self.conv(torch.cat([inputs, self.hidden_states], dim = 1))
        cc_i, cc_f, cc_o, cc_g = torch.split(concat_conv, self.hidden_channels, dim = 1)

        if checkpointing:
            self.hidden_states, self.cell_states = checkpoint(chkpt_blk, cc_i, cc_f, cc_o, cc_g, self.cell_states)
        else:
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
    
            self.cell_states = fuse_mul_add_mul(f, self.cell_states, i, g)
            self.hidden_states = o * torch.tanh(self.cell_states)
        
        return self.hidden_states 


## Convolutional Tensor-Train LSTM Module
class ConvTTLSTMCell(nn.Module):

    def __init__(self,
        # interface of the Conv-TT-LSTM 
        input_channels, hidden_channels,
        # convolutional tensor-train network
        order = 3, steps = 3, ranks = 8,
        # convolutional operations
        kernel_size = 5, bias = True):
        """
        Initialization of convolutional tensor-train LSTM cell.

        Arguments:
        ----------
        (Hyper-parameters of the input/output channels)
        input_channels:  int
            Number of input channels of the input tensor.
        hidden_channels: int
            Number of hidden/output channels of the output tensor.
        Note: the number of hidden_channels is typically equal to the one of input_channels.

        (Hyper-parameters of the convolutional tensor-train format)
        order: int
            The order of convolutional tensor-train format (i.e. the number of core tensors).
            default: 3
        steps: int
            The total number of past steps used to compute the next step.
            default: 3
        ranks: int
            The ranks of convolutional tensor-train format (where all ranks are assumed to be the same).
            default: 8

        (Hyper-parameters of the convolutional operations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 5
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True
        """
        super(ConvTTLSTMCell, self).__init__()

        ## Input/output interfaces
        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels

        ## Convolutional tensor-train network
        self.steps = steps
        self.order = order
        self.lags  = steps - order + 1

        ## Convolutional operations
        #kernel_size = utils._triple(kernel_size)
        padding     = kernel_size // 2

        Conv1d = lambda in_channels, out_channels: nn.Conv1d(
            in_channels = in_channels, out_channels = out_channels, 
            kernel_size = kernel_size, padding = padding, bias = bias)

        ## Convolutional layers
        self.layers  = nn.ModuleList()
        self.layers_ = nn.ModuleList()
        for l in range(order):
            self.layers.append(Conv1d(
                in_channels  = ranks if l < order - 1 else ranks + input_channels, 
                out_channels = ranks if l < order - 1 else 4 * hidden_channels))

            self.layers_.append(Conv1d(
                in_channels = self.lags * hidden_channels, out_channels = ranks))

    def initialize(self, inputs):
        """ 
        Initialization of the hidden/cell states of the convolutional tensor-train cell.

        Arguments:
        ----------
        inputs: 4-th order tensor of size 
            [batch_size, input_channels, height, width]
            Input tensor to the convolutional tensor-train LSTM cell.

        """
        device = inputs.device # "cpu" or "cuda"
        batch_size, _, height = inputs.size()

        # initialize both hidden and cell states to all zeros
        self.hidden_states  = [torch.zeros(batch_size, self.hidden_channels, 
            height, device = device) for t in range(self.steps)]
        self.hidden_pointer = 0 # pointing to the position to be updated

        self.cell_states = torch.zeros(batch_size, 
            self.hidden_channels, height, device = device)

    def forward(self, inputs, first_step = False, checkpointing = False):
        """
        Computation of the convolutional tensor-train LSTM cell.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, input_channels, height, width] 
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the tensor is the first step in the input sequence. 
            Note: If so, both hidden and cell states are intialized to zeros tensors.
            default: False

        checkpointing: bool
            Whether to use the checkpointing technique to reduce memory expense.
            default: True
        
        Returns:
        --------
        hidden_states: a list of 4-th order tensor of size 
            [batch_size, input_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.

        """

        if first_step: self.initialize(inputs) # intialize the states at the first step

        ## (1) Convolutional tensor-train module
        for l in range(self.order):
            input_pointer = self.hidden_pointer if l == 0 else (input_pointer + 1) % self.steps

            input_states = self.hidden_states[input_pointer:] + self.hidden_states[:input_pointer]
            input_states = input_states[:self.lags]

            input_states = torch.cat(input_states, dim = 1)
            input_states = self.layers_[l](input_states)

            if l == 0:
                temp_states = input_states
            else: # if l > 0:
                temp_states = input_states + self.layers[l-1](temp_states)
                
        ## (2) Standard convolutional-LSTM module
        concat_conv = self.layers[-1](torch.cat([inputs, temp_states], dim = 1))
        cc_i, cc_f, cc_o, cc_g = torch.split(concat_conv, self.hidden_channels, dim = 1)

        if checkpointing:
            outputs, self.cell_states = checkpoint(chkpt_blk, cc_i, cc_f, cc_o, cc_g, self.cell_states)
        else:
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
    
            self.cell_states = fuse_mul_add_mul(f, self.cell_states, i, g)
            outputs = o * torch.tanh(self.cell_states)

        self.hidden_states[self.hidden_pointer] = outputs
        self.hidden_pointer = (self.hidden_pointer + 1) % self.steps
        
        return outputs

## Convolutional-LSTM network
class ConvLSTMNet(nn.Module):
    def __init__(self,
        # input to the model
        input_channels,
        # architecture of the model
        layers_per_block, hidden_channels, skip_stride = None,
        # parameters of convolutional tensor-train layers
        cell = "convlstm", cell_params = {}, 
        # parameters of convolutional operation
        kernel_size = 3, bias = True,
        # output function and output format
        output_sigmoid = False):
        """
        Initialization of a Conv-LSTM network.
        
        Arguments:
        ----------
        (Hyper-parameters of input interface)
        input_channels: int 
            The number of channels for input video.
            Note: 3 for colored video, 1 for gray video. 

        (Hyper-parameters of model architecture)
        layers_per_block: list of ints
            Number of Conv-LSTM layers in each block. 
        hidden_channels: list of ints
            Number of output channels.
        Note: The length of hidden_channels (or layers_per_block) is equal to number of blocks.

        skip_stride: int
            The stride (in term of blocks) of the skip connections
            default: None, i.e. no skip connection
        
        [cell_params: dictionary

            order: int
                The recurrent order of convolutional tensor-train cells.
                default: 3
            steps: int
                The number of previous steps used in the recurrent cells.
                default: 5
            rank: int
                The tensor-train rank of convolutional tensor-train cells.
                default: 16
        ]
        
        (Parameters of convolutional operations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            default: 3
        bias: bool 
            Whether to add bias in the convolutional operation.
            default: True

        (Parameters of the output function)
        output_sigmoid: bool
            Whether to apply sigmoid function after the output layer.
            default: False
        """
        super(ConvLSTMNet, self).__init__()

        ## Hyperparameters
        self.layers_per_block = layers_per_block
        self.hidden_channels  = hidden_channels

        self.num_blocks = len(layers_per_block)
        assert self.num_blocks == len(hidden_channels), "Invalid number of blocks."

        self.skip_stride = (self.num_blocks + 1) if skip_stride is None else skip_stride

        self.output_sigmoid = output_sigmoid

        ## Module type of convolutional LSTM layers

        if cell == "convlstm": # standard convolutional LSTM
            Cell = lambda in_channels, out_channels: ConvLSTMCell(
            input_channels = in_channels, hidden_channels = out_channels,
            kernel_size = kernel_size, bias = bias)

        elif cell == "convttlstm": # convolutional tensor-train LSTM
            Cell = lambda in_channels, out_channels: ConvTTLSTMCell(
                input_channels = in_channels, hidden_channels = out_channels,
                order = cell_params["order"], steps = cell_params["steps"], ranks = cell_params["ranks"], 
                kernel_size = kernel_size, bias = bias)
        else:
            raise NotImplementedError

        ## Construction of convolutional tensor-train LSTM network

        # stack the convolutional-LSTM layers with skip connections 
        self.layers = nn.ModuleDict()
        for b in range(self.num_blocks):
            for l in range(layers_per_block[b]):
                # number of input channels to the current layer
                if l > 0: 
                    channels = hidden_channels[b]
                elif b == 0: # if l == 0 and b == 0:
                    channels = input_channels
                else: # if l == 0 and b > 0:
                    channels = hidden_channels[b-1]
                    if b > self.skip_stride:
                        channels += hidden_channels[b-1-self.skip_stride] 

                lid = "b{}l{}".format(b, l) # layer ID
                self.layers[lid] = Cell(channels, hidden_channels[b])

        # number of input channels to the last layer (output layer)
        channels = hidden_channels[-1]
        if self.num_blocks >= self.skip_stride:
            channels += hidden_channels[-1-self.skip_stride]

        self.layers["output"] = nn.Conv1d(channels, input_channels, 
            kernel_size = 1, padding = 0, bias = True)


    def forward(self, inputs, input_frames, future_frames, output_frames, 
        teacher_forcing = False, scheduled_sampling_ratio = 0, checkpointing = False):
        """
        Computation of Convolutional LSTM network.
        
        Arguments:
        ----------
        inputs: a 5-th order tensor of size [batch_size, input_frames, input_channels, height, width] 
            Input tensor (video) to the deep Conv-LSTM network. 
        
        input_frames: int
            The number of input frames to the model.
        future_frames: int
            The number of future frames predicted by the model.
        output_frames: int
            The number of output frames returned by the model.

        teacher_forcing: bool
            Whether the model is trained in teacher_forcing mode.
            Note 1: In test mode, teacher_forcing should be set as False.
            Note 2: If teacher_forcing mode is on,  # of frames in inputs = total_steps
                    If teacher_forcing mode is off, # of frames in inputs = input_frames
        scheduled_sampling_ratio: float between [0, 1]
            The ratio of ground-truth frames used in teacher_forcing mode.
            default: 0 (i.e. no teacher forcing effectively)

        Returns:
        --------
        outputs: a 5-th order tensor of size [batch_size, output_frames, hidden_channels, height, width]
            Output frames of the convolutional-LSTM module.
        """

        # compute the teacher forcing mask 
        if teacher_forcing and scheduled_sampling_ratio > 1e-6:
            # generate the teacher_forcing mask (4-th order)
            teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio * 
                torch.ones(inputs.size(0), future_frames - 1, 1, 1, 1, device = inputs.device))
        else: # if not teacher_forcing or scheduled_sampling_ratio < 1e-6:
            teacher_forcing = False

        # the number of time steps in the computational graph
        total_steps = input_frames + future_frames
        outputs = [None] * total_steps

        for t in range(total_steps):
            # input_: 4-th order tensor of size [batch_size, input_channels, height, width]
            if t < input_frames: 
                input_ = inputs[:, t]
            elif not teacher_forcing:
                input_ = outputs[t-1]
            else: # if t >= input_frames and teacher_forcing:
                mask = teacher_forcing_mask[:, t - input_frames]
                input_ = inputs[:, t] * mask + outputs[t-1] * (1 - mask)

            queue = [] # previous outputs for skip connection
            for b in range(self.num_blocks):
                for l in range(self.layers_per_block[b]):
                    lid = "b{}l{}".format(b, l) # layer ID
                    input_ = self.layers[lid](input_, 
                        first_step = (t == 0), checkpointing = checkpointing)

                queue.append(input_)
                if b >= self.skip_stride:
                    input_ = torch.cat([input_, queue.pop(0)], dim = 1) # concat over the channels

            # map the hidden states to predictive frames (with optional sigmoid function)
            outputs[t] = self.layers["output"](input_)
            if self.output_sigmoid:
                outputs[t] = torch.sigmoid(outputs[t])

        # return the last output_frames of the outputs
        outputs = outputs[-output_frames:]

        # 5-th order tensor of size [batch_size, output_frames, channels, height, width]
        outputs = torch.stack([outputs[t] for t in range(output_frames)], dim = 1)

        return outputs
  
class convttlstm(pl.LightningModule):
  def __init__(self,input_frames,future_frames,output_frames,cae_weights, model='convttlstm', lr=1e-3):
    super().__init__()
    self.model = ConvLSTMNet(
        input_channels = 1, 
        output_sigmoid = False,
        # model architecture
        layers_per_block = (3,4,4,3), 
        hidden_channels  = (32,48,48,32), 
        skip_stride = 2,
        # convolutional tensor-train layers
        cell = model,
        cell_params = {
            "order": 3, 
            "steps": 5, 
            "ranks": 8},
        # convolutional parameters
        kernel_size = 3)
    self.autoencoder = Autoencoder.load_from_checkpoint(cae_weights)
    self.input_frames = input_frames
    self.output_frames = output_frames
    self.future_frames = future_frames
    self.lr = lr
  
  def encode(self,x):
    latent = []
    for i in range(x.shape[1]):
        latent.append(self.autoencoder.encoder(x[:,i]).unsqueeze(1))
    return torch.cat(latent,dim=1).unsqueeze(2)
  
  def decode(self,x):
    y = []
    for i in range(x.shape[1]):
        y.append(self.autoencoder.decoder(x[:,i,0]).unsqueeze(1))
    return torch.cat(y,dim=1)

  def forward(self,x,input_frames,future_frames,output_frames,teacher_forcing=False):
      x = self.encode(x)
      pred = self.model(x[:,:input_frames], 
                  input_frames  =  input_frames, 
                  future_frames = future_frames, 
                  output_frames = output_frames, 
                  teacher_forcing = False)
      return self.decode(pred)

  def loss(self,output,target):
    result = F.l1_loss(output,target) + F.mse_loss(output,target)
    return result
  
  def training_step(self,batch,batch_idx):
    inputs = batch[:, :self.future_frames+self.input_frames]
    origin = batch[:, self.input_frames:self.future_frames+self.input_frames][:,-self.output_frames:]

    pred = self(inputs,self.input_frames,self.future_frames,self.output_frames,teacher_forcing=True)

    loss = self.loss(pred, origin)
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    inputs = batch[:,  :self.input_frames]
    origin = batch[:, self.input_frames:self.future_frames+self.input_frames][:,-self.output_frames:]
    pred = self(inputs,self.input_frames,self.future_frames,self.output_frames)
    loss = torch.sqrt(F.mse_loss(origin,pred)/torch.mean(origin**2))
    self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    return {'val_loss': loss}

  def configure_optimizers(self):
      opt =  torch.optim.Adam(self.parameters(), lr=self.lr)
      return  {
        'optimizer': opt,
        'lr_scheduler': {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=20, threshold=1e-2, min_lr=1e-5),
            'monitor': 'val_loss',
        }
    }
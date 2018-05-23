import numpy as np
import torch
import torch.nn as nn
import nn_ops
import utils                         

class DecoderBlock(nn.Module):

  def __init__(self, n_in_planes, n_out_planes):
    super().__init__()
    self.block = nn.Sequential(
      nn_ops.deconv4x4(n_in_planes, n_out_planes, True),
      nn.BatchNorm2d(n_out_planes),
      nn.ReLU(inplace=True),
      nn_ops.conv3x3(n_out_planes, n_out_planes, 1, True),
      nn.BatchNorm2d(n_out_planes)
    )

    self.upsample = lambda x: nn.functional.upsample(
      x, scale_factor=2, mode='nearest')
    self.shortcut_conv = nn.Sequential()
    if n_in_planes != n_out_planes:
      self.shortcut_conv = nn.Sequential(
        nn.Conv2d(n_in_planes, n_out_planes, kernel_size=1),
        nn.BatchNorm2d(n_out_planes)
      )

  def forward(self, x):
    out = self.block(x)
    shortcut = self.shortcut_conv(x)
    shortcut = self.upsample(shortcut)

    out += shortcut
    out = nn.functional.relu(out)
    return out

class celebA_Decoder(nn.Module):

  def __init__(self, d_latent, device='cuda', log_dir=''):
    super().__init__()

    self.d_latent = d_latent
    self.device = device

    self.mult = 8
    self.latent_mapping = nn.Sequential(
      nn.Linear(self.d_latent, 4 * 4 * 128 * self.mult),
      nn.BatchNorm1d(4 * 4 * 128 * self.mult),
      nn.ReLU()
    )
    self.block1 = DecoderBlock(128 * self.mult, 64 * self.mult)
    self.block2 = DecoderBlock(64 * self.mult, 32 * self.mult)
    self.block3 = DecoderBlock(32 * self.mult, 16 * self.mult)
    self.block4 = DecoderBlock(16 * self.mult, 8 * self.mult)
    self.output_conv = nn_ops.conv3x3(8 * self.mult, 3, 1, True)
    self.final_act = nn.Sigmoid()

    self.apply(nn_ops.variable_init)
    self.to(device)
    utils.model_info(self, 'celebA_decoder', log_dir)

  def forward(self, y):
    x = self.latent_mapping(y)
    x = x.view(-1, 128 * self.mult, 4, 4)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.output_conv(x)
    x = self.final_act(x)
    return x

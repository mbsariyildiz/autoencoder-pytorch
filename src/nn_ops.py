import torch
import torch.nn as nn

def variable_init(m, neg_slope=0.0):
  with torch.no_grad():
    if isinstance(m, (nn.Linear, nn.Conv2d)):
      nn.init.kaiming_uniform_(m.weight, neg_slope)
      if m.bias is not None:
        m.bias.zero_()
    elif isinstance(m, nn.BatchNorm2d):
      if m.weight is not None:
        m.weight.fill_(1)
      if m.bias is not None:
        m.bias.zero_()
      if m.running_mean is not None:
        m.running_mean.zero_()
      if m.running_var is not None:
        m.running_var.zero_()

def deconv4x4(n_in_planes, n_out_planes, bias=True):
  """4x4 convolution with padding"""
  return nn.ConvTranspose2d(n_in_planes, 
                           n_out_planes, 
                           kernel_size=4, 
                           stride=2,
                           padding=1,
                           bias=bias) 

def conv3x3(in_planes, out_planes, stride=1, bias=True):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=bias) 

class SigmoidNoGradient(torch.autograd.Function):

  def forward(self, x):
    return torch.nn.functional.sigmoid(x)

  def backward(self, g):
    return g.clone()

class PlusMinusOne(object):
  def __call__(self, x):
    return x  * 2.0 - 1.0
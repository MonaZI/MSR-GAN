import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.net_gen()

    def net_gen(self):
        """
        Generates the NN
        :return: Nothing is returned
        """
        in_size = self.args.mask_len
        mid_size = self.args.mid_size
        modules_tail = [spectral_norm(nn.Linear(in_size, mid_size, bias=True)),
                     nn.ReLU(),
                     spectral_norm(nn.Linear(mid_size, mid_size//2, bias=True)),
                     nn.ReLU(),
                     spectral_norm(nn.Linear(mid_size//2, 1, bias=True))]
        self.tail = nn.Sequential(*modules_tail)

        # initializing the layers
        self.tail.apply(self.weights_init)

    def forward(self, x):
        x = self.tail(x)
        return x

    def weights_init(self, m):
        """
        Initializes the weights with zero-mean normal distribution and 0.1 std
        :param m: the network
        :return: Nothing returned, parameters initialized in-place
        """
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
            m.weight.data.normal_(0.0, 0.01) #0.01 earlier, I changed it for MSR-GAN
            m.bias.data.fill_(0)


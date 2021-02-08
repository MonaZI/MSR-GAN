import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np


class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        """
        The AE network for the MMD GAN paper
        """
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.net_gen()

    def net_gen(self):
        """
        Generates the NN
        :return: Nothing is returned
        """
        in_size = self.args.mask_len
        mid_size = self.args.z_mid_size//2
        # added third layer for the new experiments
        modules_enc = [nn.Linear(in_size, in_size, bias=True),
                     nn.ReLU(),
                     nn.Linear(in_size, in_size, bias=True),
                     nn.ReLU(),
                     nn.Linear(in_size, in_size//2, bias=True),
                     nn.ReLU(),
                     nn.Linear(in_size//2, in_size//2, bias=True),
                     nn.ReLU(),
                     nn.Linear(in_size//2, in_size//4, bias=True)]

        modules_dec = [nn.Linear(in_size//4, in_size//2, bias=True),
                     nn.ReLU(),
                     nn.Linear(in_size//2, in_size//2, bias=True),
                     nn.ReLU(),
                     nn.Linear(in_size//2, in_size, bias=True),
                     nn.ReLU(),
                     nn.Linear(in_size, in_size, bias=True),
                     nn.ReLU(),
                     nn.Linear(in_size, in_size, bias=True)]
        self.encoder = nn.Sequential(*modules_enc)
        self.decoder = nn.Sequential(*modules_dec)

        # initializing the layers
        self.encoder.apply(self.weights_init)
        self.decoder.apply(self.weights_init)

    def forward(self, x, enc_only=False):
        feat = self.encoder(x)
        if enc_only:
            return feat
        x = self.decoder(feat)
        return feat, x

    def weights_init(self, m):
        """
        Initializes the weights with zero-mean normal distribution and 0.1 std
        :param m: the network
        :return: Nothing returned, parameters initialized in-place
        """
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0)

class NetGen(nn.Module):
    def __init__(self, args):
        super(NetGen, self).__init__()
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.net_gen()

    def net_gen(self):
        """
        Generates the NN
        :return: Nothing is returned
        """
        in_size = self.args.z_size
        mid_size = self.args.z_mid_size//2
        # added third layer for the new experiments
        modules_tail = [(nn.Linear(in_size, mid_size, bias=True)),
                     nn.ReLU(),
                     (nn.Linear(mid_size, mid_size*2, bias=True)),
                     nn.ReLU(),
                     (nn.Linear(mid_size*2, mid_size*4, bias=True)),
                     nn.ReLU(),
                     (nn.Linear(mid_size*4, self.args.sig_len, bias=True))]
        self.tail = nn.Sequential(*modules_tail)

        # initializing the layers
        self.tail.apply(self.weights_init)

    def forward(self, x):
        x = self.tail(x)
        x = self.sigmoid(x).clone()
        return x

    def weights_init(self, m):
        """
        Initializes the weights with zero-mean normal distribution and 0.1 std
        :param m: the network
        :return: Nothing returned, parameters initialized in-place
        """
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.net_gen()

    def net_block(self, n):
        """
        Generates a block which is used in the network
        :param n: a parameter controling the number of the channels
        :return: a NN block
        """
        if n==1:
            in_channels = 1
        else:
            in_channels = (n-1)*self.args.num_channels
        return [nn.Conv1d(in_channels, n*self.args.num_channels, self.args.kernel_size, padding=(self.args.kernel_size//2), bias=True),
                 self.maxPooling,
                 nn.ReLU()]

    def net_gen(self):
        """
        Generates the NN
        :return: Nothing is returned
        """
        # if you have a continuous signal and you want to have convolution layers uncomment the following lines
        #self.maxPooling = nn.MaxPool1d(2, stride=2)
        #modules_body = []
        #for i in range(self.args.num_blocks):
        #    modules_body.extend(self.net_block(i+1))
        #self.body = nn.Sequential(*modules_body)
        #in_size = (self.args.mask_len // (2**self.args.num_blocks))*self.args.num_blocks*self.args.num_channels
        in_size = self.args.mask_len
        mid_size = self.args.mid_size
        # added third layer for the new experiments
        modules_tail = [spectral_norm(nn.Linear(in_size, mid_size, bias=True)),
                     nn.ReLU(),
                     spectral_norm(nn.Linear(mid_size, mid_size//2, bias=True)),
                     nn.ReLU(),
                     spectral_norm(nn.Linear(mid_size//2, 1, bias=True))]
        self.tail = nn.Sequential(*modules_tail)

        # initializing the layers
        #self.body.apply(self.weights_init)
        self.tail.apply(self.weights_init)

    def forward(self, x):
        #x = self.body(x)
        #x = x.view([x.shape[0], x.shape[1]*x.shape[2]])
        x = self.tail(x)
        # MONA: Uncomment below for your encoder based model
        #x = self.sigmoid(x).clone()
        #x -= 0.5
        #x *= 2.
        #x *= 4.
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


class pdfNet(nn.Module):
    def __init__(self, args):
        super(pdfNet, self).__init__()
        self.args = args
        self.net_gen()
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

    def net_gen(self):
        """
        Generates the network based on the given arguments
        :return:
        """
        in_size = self.args.z_size
        mid_size = self.args.z_mid_size
        modules_tail = [nn.Linear(in_size, mid_size, bias=True),
                     nn.ReLU(),
                     nn.Linear(mid_size, mid_size//2, bias=True),
                     nn.ReLU(),
                     nn.Linear(mid_size//2, 2, bias=True)]
        self.tail = nn.Sequential(*modules_tail)
        self.tail.apply(self.weights_init)

    def forward(self, z):
        #res = self.sigmoid(self.tail(z)).clone()
        res = self.tail(z)
        res = torch.atan(res[:, 0]/res[:, 1])
        res /= (np.pi/2)
        #res -= 0.5
        #res *= 2
        res *= (self.args.supp-self.args.num_coeff-1.)
        #p = z
        #for b in self.tail:
        #    p = b(p)

        ##p = self.tail(z)
        ##import pdb; pdb.set_trace()
        #p = self.softmax(p)
        ## this is to combine the 1-hot representation into a single number
        #tmp = torch.arange(0, self.args.sig_len).unsqueeze(0).repeat(self.args.batch_size, 1).float().cuda()
        #shifts_max = torch.argmax(p, dim=1)
        #tt = (p==torch.max(p, dim=1)[0].unsqueeze(1).repeat(1, self.args.sig_len)).float()
        #res = torch.sum(tt*p, dim=1)
        ##p = torch.sum(p * tmp, dim=1)
        ## shifts_max = torch.floor(p*(self.args.sig_len-1)).squeeze()
        return res

    def weights_init(self, m):
        """
        Initializes the weights with zero-mean normal distribution and 0.1 std
        :param m: the network
        :return: Nothing returned, parameters initialized in-place
        """
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0)

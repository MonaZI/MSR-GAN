# the alternating training between the signal and the discriminator network
import os
from abc import abstractmethod

import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt

from utils import * 

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class TrainerAbstract(object):
    def __init__(self, net, meas, dataloader, args):
        """
        The initialization for the Trainer object
        :param net: the network
        :param meas: the set of real measurements
        :param dataloader: the dataloader to load the original measurements
        :param args: a set of arguments
        """
        self.Softmax = torch.nn.Softmax(dim=0)
        self.args = args
        self.meas = meas
        self.dataloader = dataloader
        self.pdf_unif = np.ones(self.args.pdf_vec.shape)/len(self.args.pdf_vec)
        if self.args.use_gpu:
            self.net = net.cuda()
        self.logger_tf = SummaryWriter(log_dir=os.path.join(self.args.log_path, self.args.exp_name))

        # initialization of the signal and the pdf
        self.x = (torch.rand((self.args.sig_len,))) 
        self.x = self.x.requires_grad_()
        self.p = torch.zeros((self.args.sig_len,))
        self.pdf = self.Softmax(self.p)

        self.recon_x = np.zeros((self.args.sig_len, self.args.num_epoch//self.args.iterLog))
        self.recon_pdf = np.zeros((self.args.sig_len, self.args.num_epoch//self.args.iterLog))

        # generate two different optimizers for the variables and the discriminator network weights
        self.optim_net = optim.SGD(self.net.parameters(), lr=self.args.lrate, weight_decay=self.args.wdecay, momentum=0.9)
        self.iteration = 0

    def adjust_lrate(self):
        """
        Changes the learning rate after some iterations
        :return: nothing is returned
        """
        for param_group in self.optim_net.param_groups:
            param_group['lr'] *= self.args.gamma_lrate
        for param_group in self.optim_x.param_groups:
            param_group['lr'] *= self.args.gamma_lrate
        self.args.lrate_pdf *= self.args.gamma_lrate

    def train(self, x_true, p_true, meas):
        """
        Trains and saves the trained model
        :param x_true: the gt signal, only used for comparison with the recon signal
        :param p_true: the gt pdf, only used for comparison with the recon pdf
        :param meas: the set of noisy partial measurements
        :return: nothing is returned
        """
        dict = {}
        if self.args.correct_pdf:
            self.pdf = torch.tensor(p_true)
        #dict['init_x'] = self.x.detach().cpu().numpy()
        dict['init_p'] = self.pdf.detach().cpu().numpy()
        #self.train_epoch(x_true, alpha_true, p_true)
        self.train_epoch(x_true, p_true)
        print('Finished training!')
        
        dict['recon_x'] = self.recon_x
        dict['gt_x'] = x_true
        dict['gt_pdf'] = p_true
        dict['meas'] = meas
        dict['sigma'] = self.args.sigma
        if not self.args.correct_pdf and not self.args.unif_pdf:
            dict['recon_pdf'] = self.recon_pdf
        return self.x.sig.detach().cpu().numpy()

    @abstractmethod
    def train_epoch(self, x_true, p_true):
        pass

    def log(self):
        """
        Logs the current status of the model on val and test splits
        :return: Nothing
        """
        for tag, value in self.net.named_parameters():
            tag = tag.replace('.', '/')
            self.logger_tf.add_histogram(tag, value.data.cpu().numpy(), self.iteration)
            if not(value.grad is None):
                self.logger_tf.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), self.iteration)

        if  not(self.x.sig.grad is None):
            self.logger_tf.add_histogram('grad_x', self.x.sig.grad.data.cpu().numpy(), self.iteration)
        

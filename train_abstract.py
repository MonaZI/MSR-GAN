# the alternating training between the signal and the discriminator network
import os
from abc import abstractmethod

import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt

from utils import meas_gen_pdf, meas_gen_pdf_tensor, find_dist_pdf

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
        self.logger_tf = SummaryWriter(log_dir=os.path.join(self.args.logPath, self.args.expName))

        # initialization of the signal and the pdf
        self.x = (torch.rand((self.args.sig_len,))) #*0.01 # remove 0.01
        #self.x = torch.zeros((self.args.sig_len, ))
        self.x = self.x.requires_grad_() # ADD CUDA BACK
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
            param_group['lr'] *= self.args.gamma_lrate #0.92
        #for param_group in self.optim_encoder.param_groups:
        #    param_group['lr'] *= self.args.gamma_lrate #0.92
        #for param_group in self.optim_x.param_groups:
        #    param_group['lr'] *= self.args.gamma_lrate 
        #if not self.args.correct_pdf:
        #    for param_group in self.optim_pdf.param_groups:
        #        param_group['lr'] *= self.args.gamma_lrate #0.92


    def train(self, x_true, p_true, meas): # alpha_true after x_true
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
        torch.save(self.net.state_dict(), os.path.join(self.args.modelSavePath, self.args.expName))
        dict['recon_x'] = self.recon_x
        dict['gt_x'] = x_true
        dict['gt_pdf'] = p_true
        dict['meas'] = meas
        dict['sigma'] = self.args.sigma
        if not self.args.correct_pdf and not self.args.unif_pdf:
            dict['recon_pdf'] = self.recon_pdf
        scipy.io.savemat(self.args.modelSavePath + '/' + self.args.expName + '_results.mat', dict, do_compression=True)
        return self.x.detach().cpu().numpy()

    @abstractmethod
    def train_epoch(self, x_true, p_true):
        pass

    def pdf_update(self, real_meas):
        """
        Updating the pdf based on the current values of x
        :param real_meas: the batch of the real measurements
        """
        # find the assignment of each of the measurements to a masked cluster
        sig = self.x.detach().cpu().numpy()
        clusters = np.zeros((self.args.sig_len, self.args.mask_len))
        for i in range(self.args.sig_len):
            # the circular shift of the signal
            tmp = np.concatenate((sig[i:], sig[0:i]), axis=0)
            tmp = tmp[0:self.args.mask_len]
            clusters[i, :] = tmp

        # assignment of each measurement to a cluster
        dist = np.zeros((self.args.batch_size, self.args.sig_len))
        meas = real_meas.cpu().numpy()
        for m in range(self.args.batch_size):
            for c in range(self.args.sig_len):
                dist[m, c] = self.pdf[c] * np.exp(-1.*np.linalg.norm(meas[m,:]-clusters[c,:])**2 / (2.*self.args.sigma**2)) + 1e-7
            dist[m,:] = dist[m,:]/np.sum(dist[m,:])
        # now updating p based on the assignments
        self.pdf = np.mean(dist, axis=0)
        self.pdf /= np.sum(self.pdf)

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

        #for tag, value in self.net.named_parameters():
        #    tag = tag.replace('.', '/')
        #    self.logger_tf.add_histogram(tag+'_x', value.data.cpu().numpy(), self.iteration)
        #    self.logger_tf.add_histogram(tag + '/grad_x', value.grad.data.cpu().numpy(), self.iteration)
         
                    
        if  not(self.x.sig.grad is None):
            self.logger_tf.add_histogram('grad_x', self.x.sig.grad.data.cpu().numpy(), self.iteration)
        #
        #if not self.args.correct_pdf:
        #    for tag, value in self.net_theta.named_parameters():
        #        tag = tag.replace('.', '/')
        #        self.logger_tf.add_histogram(tag+'_theta', value.data.cpu().numpy(), self.iteration)
        #        if not(value.grad is None):
        #            self.logger_tf.add_histogram(tag + '/grad_theta', value.grad.data.cpu().numpy(), self.iteration)


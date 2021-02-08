import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from train_abstract import TrainerAbstract
from utils import *


class signalClass(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        init = (torch.rand((self.args.sig_len,)) - 0.5) * 1.
        self.sig = nn.Parameter(init, requires_grad=True)

    def forward(self, shifts):
        y = meas_gen_shifts_tensor(self.sig, self.args.mask_len, shifts, sigma=self.args.sigma).float().cuda()
        return y

class TrainerWithPDFGumbel(TrainerAbstract):
    """
    Trainer with training also for the pdf, using the gumbel-softmax ideas
    """
    def __init__(self, net, meas, dataloader, args):
        TrainerAbstract.__init__(self, net, meas, dataloader, args)
        self.x = signalClass(self.args)
        self.optim_x = optim.SGD(self.x.parameters(), lr=self.args.lrate_x, weight_decay=self.args.wdecay, momentum=0.9)
        if self.args.periodic:
            self.a = torch.rand(self.args.a_size,).float()
            self.a.requires_grad = True
            self.x = sig_from_a(self.a, self.args.sig_len)

    def train_epoch(self, x_true, p_true):
        """
        Trains the discriminator and reconstructs the signal
        :param x_true: the gt signal, only used for comparison with the recon signal
        :param p_true: the gt pdf, only used for comparison with the recon pdf
        :return: nothing is returned
        """
        for epoch in range(self.args.num_epoch):
            # updating the discriminator
            # freeze everything else except for the discriminator
            self.p = self.p.detach().cpu()
            self.pdf = self.pdf.detach().cpu()

            for param in self.net.parameters():
                param.requires_grad = True

            for param in self.x.parameters():
                param.requires_grad = False

            for iter, real_meas in enumerate(self.dataloader):
                real_meas = real_meas.cuda()
                self.optim_net.zero_grad()
                if iter==self.args.n_disc:
                    break

                # generating synthetic measurements
                if not self.args.correct_pdf:
                    shift_probs, shifts = gumbel_softmax_sampler(self.pdf, self.args.batch_size, self.args.tau)
                else:
                    if self.args.correct_pdf:
                        # sample from the correct pdf
                        shifts = meas_shifts_pdf_msr(self.args.batch_size, self.args.pdf_vec, self.args) 
                    if self.args.unif_pdf:
                        # sample from the uniform pdf
                        shifts = meas_shifts_pdf_msr(self.args.batch_size, self.pdf_unif, self.args) 
                syn_meas = self.x.forward(shifts) 
                loss_syn = torch.mean(self.net(syn_meas))
                loss_real = torch.mean(self.net(real_meas))

                # interpolation term
                if (self.args.lamb>0.):
                    alpha = torch.rand((self.args.batch_size, 1, 1)).float().cuda()
                    int_meas = alpha * real_meas + (1-alpha) * syn_meas
                    int_meas.requires_grad = True
                    out = self.net(int_meas)
                    gradients = torch.autograd.grad(outputs=out, inputs=int_meas, grad_outputs=torch.ones(out.shape).cuda())[0].squeeze()
                    reg = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                else:
                    reg = 0.

                loss = -1*(loss_real - loss_syn - self.args.lamb * reg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                self.optim_net.step()
                self.logger_tf.add_scalar('loss-disc', loss.data.cpu().numpy().item(), self.iteration)

                self.iteration += 1

                if (self.iteration%self.args.iterLog==0):
                    self.log()
                    # save the results
                    self.recon_x[:, epoch//self.args.iterLog] = self.x.sig.detach().cpu().numpy()
                    if not self.args.correct_pdf and not self.args.unif_pdf:
                        self.recon_pdf[:, epoch//self.args.iterLog] = self.pdf.detach().cpu().numpy()
                    savedict = {}
                    savedict['recon_x'] = self.recon_x
                    savedict['recon_pdf'] = self.recon_pdf
                    savedict['gt_x'] = x_true
                    savedict['gt_pdf'] = p_true
                    scipy.io.savemat(savedict, os.path.join('./results', self.args.expName + '.mat'))

            if epoch%2000==0:
                self.args.tau = max(0.5, np.exp(-3e-5*epoch))

            # updating the signal
            loss_x = self.update_signal(epoch)

            if (epoch%self.args.iterLog==0):
                x_pred = self.x.sig.detach().cpu().numpy()
                p_pred = self.pdf.detach().numpy()

                figure, sig_aligned = find_dist_pdf(x_pred, x_true,  p_pred, p_true, self.args.expName)
                err_sig = np.linalg.norm(x_true-sig_aligned)/np.linalg.norm(x_true)
                plt.title('epoch=%d' %epoch)
                self.logger_tf.add_figure(str(epoch), figure, global_step=epoch)
                plt.close()
                print('Figure printed!')
                self.logger_tf.add_scalar('MSE', err_sig, epoch)

            self.logger_tf.add_scalar('loss-x', loss_x.data.cpu().numpy().item(), epoch)
            if (epoch%1 == 0):
                print('epoch=%d, loss_x=%f, loss_net=%f' %(epoch, loss_x.detach().cpu().numpy().item(), loss.detach().cpu().numpy().item()))

            # changing the learning rate
            if (epoch%self.args.iterChangeLR==0):
                self.adjust_lrate()

    def update_signal(self, epoch):
        """
        Updates the signal and the pdf
        :return: the loss, the max gradients over the signal and the pdf
        """
        # disable the computation of the grad wrt the discriminator network
        for param in self.net.parameters():
            param.requires_grad = False

        for param in self.x.parameters():
            param.requires_grad = True

        if (not self.args.unif_pdf) and (not self.args.correct_pdf):
            self.p.requires_grad = True
            self.pdf = self.Softmax(self.p).float().cuda()

            # sample from the distribution
            shift_probs, _ = gumbel_softmax_sampler(self.pdf, self.args.batch_size, self.args.tau)
            shifts = torch.arange(0, self.args.sig_len)

            syn_meas = self.x.forward(shifts) 
            out = self.net(syn_meas).squeeze()
            loss = -1 * torch.mean(shift_probs*out)*self.args.sig_len
        else:
            if self.args.correct_pdf:
                shifts = meas_shifts_pdf_msr(self.args.batch_size, self.args.pdf_vec, self.args) 
            if self.args.unif_pdf:
                shifts = meas_shifts_pdf_msr(self.args.batch_size, self.pdf_unif, self.args) 
            syn_meas = self.x.forward(shifts) 
            loss = -1 * torch.mean(self.net(syn_meas))
            
        self.optim_x.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.x.parameters(), 5)
        self.optim_x.step()
        
        if (not self.args.unif_pdf) and (not self.args.correct_pdf):
            grad_p = torch.autograd.grad(loss, self.p)[0]
            grad_p /= torch.norm(grad_p)
            self.p = self.p - self.args.lrate_pdf * grad_p
            self.pdf = self.Softmax(self.p)
        return loss

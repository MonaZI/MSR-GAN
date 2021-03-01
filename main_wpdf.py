import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from dataloader import Dataset, get_loader

from torch import cuda
import torch

from model import Net
from trainer_gumbel import TrainerWithPDFGumbel


def main(args):
    random_seed(args.seed)
    sig, pdf = sig_gen(args.sig_len, args.sigma, mode_sig=args.mode_sig, mode_pdf=args.mode_pdf)
    if args.pdf_periodic:
        pdf = sig_from_a(np.random.uniform(size=(args.a_size,))-0.5, args.sig_len)
        pdf -= np.min(pdf)
        pdf += 0.3
        pdf /= np.sum(pdf)
    
    args.pdf_vec = pdf
    if not os.path.isdir('./results/'):
        os.mkdir('./results/')
    meas = meas_gen_pdf(sig, args.mask_len, args.num_meas, pdf, sigma=0.)
    if args.snr=='inf':
        args.sigma=0
    else:
        args.sigma = np.sqrt(np.var(meas)/float(args.snr))
    meas += args.sigma * np.random.normal(size=meas.shape)
    dataset = Dataset(meas, args)
    dataloader = get_loader(dataset, args, is_test=False)

    # setting up the usage of GPU
    args.use_gpu = (args.gpu is not None)
    if args.use_gpu: cuda.set_device(args.gpu)
    print(args)

    net = Net(args)
    trainer = TrainerWithPDFGumbel(net, meas, dataloader, args)
    sig_pred = trainer.train(sig, pdf, meas)
    print("Finished!")


def arg_parse():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('-mode_sig', type=str, default='random', help='the type of the signal')
    parser.add_argument('-sig_len', type=int, default=8, help='the length of the signal')
    parser.add_argument('-mask_len', type=int, default=8, help='the length of the mask')
    parser.add_argument('-num_meas', type=int, default=20000, help='the number of measurements')
    parser.add_argument('-batch_size', type=int, default=200, help='the batch size')
    parser.add_argument('-sigma', type=float, default=0., help='the variance of the noise')
    parser.add_argument('-snr', default='inf', help='the snr of the projection data')
    parser.add_argument('-correct_pdf', action='store_true', default=False, help='to fix the pdf with the gt pdf')
    # model
    parser.add_argument('-num_epoch', type=int, default=50000, help='the number of epochs used for training')
    parser.add_argument('-mid_size', type=int, default=1000, help='the size of the first fully connected layer')
    parser.add_argument('-a_size', type=int, default=3, help='the size of the random vectors as the coeffs of the signal')
    parser.add_argument('-tau', type=float, default=0.5, help='temperature factor for Gumbel-Softmax')
    # optimization
    parser.add_argument('-n_disc', type=int, default=4, help='number of iteratiosn of training the discriminator')
    parser.add_argument('-lrate', type=float, default=8e-3, help='the learning rate')
    parser.add_argument('-gamma_lrate', type=float, default=0.9, help='the learning rate')
    parser.add_argument('-lrate_x', type=float, default=2e-3, help='the learning rate for x')
    parser.add_argument('-lrate_pdf', type=float, default=1e-3, help='the learning rate for the encoder')
    parser.add_argument('-wdecay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('-iterChangeLR', type=int, default=1000, help='the number of iterations after which the lrate changes')
    parser.add_argument('-iterLog', type=int, default=100, help='the number of iterations after which we log the status of the training')
    parser.add_argument('-gamma', type=float, default=0.1, help='the rate for the learning rate decay')
    parser.add_argument('-lamb', type=float, default=0., help='the weight of the regularization term')
    # misc
    parser.add_argument('-gumbel', action='store_true', default=False, help='use the gumbel method or not')
    parser.add_argument('-mode_pdf', type=str, default='nonuniform', help='mode of pdf')
    parser.add_argument('-unif_pdf', action='store_true', default=False, help='fix the pdf with uniform pdf')
    parser.add_argument('-pdf_periodic', action='store_true', default=False, help='if the pdf is periodic or not')
    parser.add_argument('-debug', action='store_true', default=False, help='to be in debug mode or not')
    parser.add_argument('-seed', type=int, default=0, help='the seed of the random generators')
    parser.add_argument('-gpu', type=int, default=0, help='the index of the gpu')
    parser.add_argument('-exp_name', type=str, default='exp', help='the name of the experiment')
    parser.add_argument('-log_path', type=str, default='./log/', help='the path to the log files')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = arg_parse()
    main(args)

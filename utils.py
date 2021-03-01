import os
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


def meas_shifts_pdf_msr(num_meas, pdf, args):
    """
    Generates the set of shifts based on the given pdf
    This function works with numpy sig.
    :param sig: the signal
    :param m: the mask length
    :param num_meas: the number of measurements
    :param pdf: the pdf of the shifts
    :param sigma: the std of noise
    :return: the set of measurements
    """
    meas = np.zeros((num_meas, ))
    cumsum = np.cumsum(pdf)
    for i in range(num_meas):
        s = np.random.rand(1)
        t = int(np.sum((cumsum<s)))
        meas[i] = t
    meas = torch.from_numpy(meas)
    return meas


def gumbel_softmax_sampler(pdf, num_meas, tau):
    """
    Draws random samples following pdf distribution
    :param pdf: the pdf
    :param num_meas: the number of samples
    :param tau: the temperature factor
    :return: the set of samples and their softmax approximation
    """
    shifts = torch.zeros(size=(num_meas,), dtype=torch.int)
    shift_probs = torch.zeros(size=(num_meas, len(pdf)))
    if pdf.is_cuda:
        shifts = shifts.cuda()
        shift_probs = shift_probs.cuda()
    g = -torch.log(-torch.log(torch.rand(size=(num_meas, len(pdf)))))
    if pdf.is_cuda: g = g.cuda()
    shifts = torch.argmax(torch.log(pdf)+g, dim=1).int().squeeze()
    tmp = torch.exp((torch.log(pdf)+g)/tau)/torch.sum(torch.exp((torch.log(pdf)+g)/tau), dim=1).unsqueeze(1)
    shift_probs = tmp

    return shift_probs, shifts


def sig_from_a(a, sig_len):
    """
    Computes the periodic signal from coefficients a
    :param a: the coefficients
    :param sig_len: the length of the signal
    :return: the signal (either a torch tensor or a numpy array)
    """
    if torch.is_tensor(a):
        t = torch.arange(0., sig_len)
        x = torch.zeros((sig_len, ))
        for count, ak in enumerate(a):
            x = x + ak * torch.sin((2*math.pi*count*t)/sig_len)
    else:
        t = np.arange(0., sig_len)
        x = np.zeros((sig_len, ))
        for count, ak in enumerate(a):
            x = x + ak * np.sin((2*math.pi*count*t)/sig_len)
    return x


def random_seed(seed):
    """
    Fixes the random seed across all random generators
    :param seed: the seed
    :return: nothing is returned
    """
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def sig_gen(d, sigma, mode_sig='random', mode_pdf='nonuniform'):
    """
    Generates a signal
    :param d: the length of the signal
    :param mode: to generate whether a random/deterministic signal
    :return: a vector representing the signal
    """
    # generating the signal
    if mode_sig=='random':
        sig = np.random.uniform(0, 1, [d,]) - 0.5
    elif mode_sig=='tri':
        sig = np.zeros(d,)
        sig[0:d//2] = np.linspace(0, d//2, d//2) / (d//2)
    else:
        tmp = np.linspace(0, 2*math.pi, d)
        sig = 2*np.cos(3*tmp)+2*np.sin(tmp)
        sig /= np.max(np.abs(sig))

    # generating the pdf
    if mode_pdf=="nonuniform":
        pdf = np.random.uniform(0, 1, [d,])
        pdf /= np.sum(pdf)
    elif mode_pdf=="uniform":
        pdf = np.ones((d,)) / d
    else:
        tmp = np.linspace(0, 2*math.pi, d)
        pdf = (np.cos(3*tmp) * np.exp(-0.1*(tmp**2)))
        pdf = pdf - np.min(pdf) + 0.5
        pdf /= np.sum(pdf)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(sig, label='sig')
    axes[0].plot(sig + sigma*np.random.normal(size=sig.shape), label='noisy sig')
    axes[0].set_ylabel('sig')
    axes[0].legend()
    axes[1].plot(pdf)
    axes[1].set_ylabel('pdf')
    plt.savefig('input.png')
    plt.close()
    print('Figure printed!')

    return sig, pdf


def meas_gen_pdf(sig, m, num_meas, pdf, sigma=0.):
    """
    Generates the set of measurements from the signal sig based on a pdf
    This function works with numpy sig.
    :param sig: the signal
    :param m: the mask length
    :param num_meas: the number of measurements
    :param pdf: the pdf of the shifts
    :param sigma: the std of noise
    :return: the set of measurements
    """
    meas = np.zeros((num_meas, m))
    cumsum = np.cumsum(pdf)
    for i in range(num_meas):
        s = np.random.rand(1)
        t = int(np.sum((cumsum<s)))
        tmp = np.concatenate((sig[t:], sig[0:t]), axis=0)
        tmp = tmp[0:m]
        meas[i,:] = tmp + sigma * np.random.normal(size=tmp.shape)
    return meas


def meas_gen_shifts(sig, m, shifts, sigma=0.):
    """
    Generates the set of measurements from the signal sig based the input shifts
    This function works with numpy sig.
    :param sig: the signal
    :param m: the mask length
    :param shifts: the shifts used to generate the measurements
    :param sigma: the std of noise
    :return: the set of measurements
    """
    num_meas = len(shifts)
    meas = np.zeros((num_meas, m))
    for i in range(num_meas):
        s = shifts[i]
        tmp = np.concatenate((sig[s:], sig[0:s]), axis=0)
        tmp = tmp[0:m]
        meas[i,:] = tmp + sigma * np.random.normal(size=tmp.shape)
    return meas


def meas_gen_shifts_tensor(sig, m, shifts, sigma=0.):
    """
    Generates the set of measurements from the signal sig based on the input shifts
    This function works with the tensor sig. Remember you had to separate this function and the one below due to the
    differentiation problems.
    :param sig: the signal
    :param m: the mask length
    :param dist: the distribution from which the measurements are generated
    :return: the set of measurements
    """
    num_meas = len(shifts)
    meas = torch.zeros((num_meas, m))
    mat = torch.zeros((len(sig), m))
    for j in range(len(sig)):
        mat[j, :] = torch.cat((sig[j:], sig[0:j]), dim=0)[0:m]

    meas = mat[np.ndarray.tolist(shifts.numpy()),:] + sigma * torch.randn(meas.shape)
    return meas


def find_dist_pdf(sig, sig_ref, pdf, pdf_ref, exp_name):
    """
    Plots the two signals and the two pdfs and compare them to one another
    :param sig: the estimated signal
    :param sig_ref: the reference gt signal
    :param pdf: the estimated pdf
    :param pdf_ref: the reference pdf
    :return: none
    """
    sig_aligned, index = align_to_ref(sig, sig_ref)
    pdf_aligned = np.concatenate((pdf[index:], pdf[0:index]), axis=0)

    figure, axes = plt.subplots(1,2, figsize=(10, 3))
    axes[0].plot(sig_aligned, label='predicted')
    axes[0].plot(sig_ref, label='gt.')
    axes[0].legend()

    axes[1].plot(pdf_aligned, label='pdf-pred')
    axes[1].plot(pdf_ref, label='pdf-gt.')
    axes[1].legend()
    if not(os.path.exists(os.path.join('./results/', exp_name))):
        os.mkdir(os.path.join('./results/', exp_name))
 
    plt.savefig('./results/'+exp_name+'/compare.png')
    return figure, sig_aligned


def relative_error(x, x_ref):
    x_aligned, _ = align_to_ref(x, x_ref)
    return np.linalg.norm(x_ref-x_aligned)/np.linalg.norm(x_ref)


def align_to_ref(sig, sig_ref):
    """
    Aligns the signal to sig_ref
    :param sig: the signal
    :param sig_ref: the reference signal
    :return: the aligned signal and the shift required for alignment of the two
    """
    # align the ref signal and the recovered one
    res = -1*float('inf')
    for s in range(len(sig)):
        tmp = np.concatenate((sig[s:], sig[0:s]), axis=0)
        inner_prod = np.sum(tmp*sig_ref)
        if inner_prod>res:
            index = s
            res = inner_prod
    sig_aligned = np.concatenate((sig[index:], sig[0:index]), axis=0)
    return sig_aligned, index

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


def meas_gen_shift_tensor(alpha, shifts, args):
    """
    Generates different shifted versions of the signals
    :param alpha: the weights of the gaussians
    :param shifts: the shifts corresponding to the measurements
    :param args: various arguments
    :return: tensor containing shifted versions of the signal
    Checked!
    """
    
    # this is for the fourier series model
    if args.cos_model:
        grid = torch.linspace(0., args.supp, args.sig_len).cuda()
        grid = grid.unsqueeze(0)-shifts.unsqueeze(1)
        tmp = torch.arange(0, args.num_coeff//2+1).unsqueeze(0).unsqueeze(0).cuda()
        tmp1 = (2*np.pi*grid.unsqueeze(2) * tmp/args.supp)
        meas = torch.sum((torch.cos(tmp1) * (alpha[0:args.num_coeff//2+1].unsqueeze(0).unsqueeze(0))), dim=2)
        meas += torch.sum((torch.sin(tmp1[:, :, 1:]) * (alpha[1+args.num_coeff//2:].unsqueeze(0).unsqueeze(0))), dim=2)
    else:
        # this is for the Gaussian models
        grid = torch.linspace(-args.supp, args.supp, args.sig_len).cuda()
        meas = torch.zeros((args.batch_size, len(grid))).cuda()
        tmp = torch.arange(0, args.num_coeff+1).unsqueeze(0).cuda()
        tmp1 = grid.unsqueeze(1)+2*tmp-args.num_coeff
        tmp2 = tmp1.unsqueeze(0)-shifts.unsqueeze(1).unsqueeze(2)
        meas = torch.sum(torch.exp(-tmp2**2/args.sigma_gauss**2) * (alpha).unsqueeze(0).unsqueeze(0), dim=2)

    #meas = meas[:, 25:25+args.mask_len]
    #for m in range(args.batch_size):
    #    #for k in range(-args.num_coeff, args.num_coeff):
    #    #    meas[m, :] += alpha[k+args.num_coeff] * torch.exp(-(grid-k-shifts[m])**2/args.sigma_gauss**2).cuda()
    #    #for k in range(0, args.num_coeff+1):
    #    #    meas[m, :] += alpha[k] * torch.exp(-(grid+2*k-args.num_coeff-shifts[m])**2/args.sigma_gauss**2)
    #    #import pdb; pdb.set_trace()
    #    meas[m, :] = torch.sum(alpha * torch.exp(-(tmp1-shifts[m])**2/args.sigma_gauss**2), dim=1)
    #import pdb; pdb.set_trace()
    meas = meas[:, 0:args.mask_len]
    return meas


def meas_shifts_all(pdf, args):
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
    meas = np.zeros((len(pdf), ))
    cumsum = np.cumsum(pdf)
    for i in range(len(pdf)):
        if not args.cos_model:
            meas[i] = (i/(len(pdf)-1) -0.5)*2.*(args.supp-args.num_coeff-1.)
        else:
            meas[i] = (t-(len(pdf)/2.)) * args.pixel_size
    meas = torch.from_numpy(meas).cuda()
    return meas


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

def meas_shifts_pdf(num_meas, pdf, args):
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
        #meas[i] = t*(2*args.supp/args.sig_len)
        if not args.cos_model:
            meas[i] = (t/(len(pdf)-1) - 0.5)*2.*(args.supp-args.num_coeff-1.)
        else:
            meas[i] = (t-(len(pdf)/2.)) * args.pixel_size
    meas = torch.from_numpy(meas).cuda()
    return meas


def gauss_sig(x, sigma):
    return np.exp(-np.abs(x)**2/(sigma**2))
    

def sig_generate(alpha, args):
    """
    Generates the signal based on its coefficients alpha
    :param alpha: the coefficients of the signal
    :param args: a set of arguments
    :return: signal and the pdf expression
    Checked!s
    """
    # K and N < T
    if args.cos_model:
        grid = np.linspace(0., args.supp, args.sig_len)
        sig = np.zeros((len(grid), ))
        for k in range(0, args.num_coeff+1):
            if k<=args.num_coeff//2:
                sig += alpha[k] * np.cos(2.*np.pi*k/args.supp*grid)
            else:
                sig += alpha[k] * np.sin(2.*np.pi*(k-args.num_coeff//2)/args.supp*grid)
    else:
        grid = np.linspace(-args.supp, args.supp, args.sig_len)
        sig = np.zeros((len(grid), ))
        for k in range(0, args.num_coeff+1):
            sig += alpha[k] * gauss_sig(grid+2*k-args.num_coeff, args.sigma_gauss)
        
    #pdf = sig_from_a(np.random.uniform(size=(args.a_size,))-0.5, args.sig_len)
    grid = np.arange(0, args.sig_len)-(args.sig_len/2.)
    pdf = (np.exp(-(grid-args.sig_len/4.)**2/50)) + np.exp(-(grid)**2/50.) + np.exp(-(grid+args.sig_len/4.)**2/50.)
    #pdf = (np.exp(-(grid)**2/200.))
    #pdf = np.exp(-(grid+args.sig_len/4.)**2/128.) + np.exp(-(grid-args.sig_len/4.)**2/128.)

    #pdf = np.exp(-(grid)**2/50.)
    #pdf = np.abs(grid)
    #pdf[pdf>(args.sig_len//4)] = 0.
    #pdf = np.exp(-np.abs(grid)/10.)

    #pdf -= np.min(pdf)
    #pdf += 0.3
    #pdf = np.zeros(grid.shape)
    #pdf[args.sig_len//4] = 1.
    #pdf[3*args.sig_len//4] = 1.
    #pdf[int(args.sig_len/2.5)] = 1.
    #pdf[args.sig_len//4:int(3*args.sig_len//4)] = 1
    #pdf[args.sig_len-args.sig_len//3:args.sig_len-args.sig_len//4] = 1
    pdf /= np.sum(pdf)
    return sig, pdf


def gen_moment_meas(meas):
    num_meas = meas.shape[0]
    moment = np.dot(np.transpose(meas), meas) / num_meas
    return moment


def gen_moment(x, pdf, mask_len):
    moment = torch.zeros((mask_len, mask_len))
    if x.is_cuda:
        moment = moment.cuda()
    for i in range(mask_len):
        for j in range(mask_len):
            tmp1 = torch.cat((x[i:],x[0:i]))
            tmp2 = torch.cat((x[j:],x[0:j]))
            moment[i, j] = torch.sum((tmp1*tmp2)*pdf)
    return moment


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
    # optimized
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
        # the following is done to make it differentiable at the borders
        #sig = (np.cos(3*tmp) * np.exp(-0.1*(tmp**2)))
        #sig = np.concatenate((sig, sig[::-1]), axis=0)
        sig = 2*np.cos(3*tmp)+2*np.sin(tmp)
        #sig = np.exp(-0.1*(tmp**2))
        #sig = tmp**3 - 1*tmp**1.5 - 1*tmp
        sig /= np.max(np.abs(sig))

    # generating the pdf
    if mode_pdf=="nonuniform":
        pdf = np.random.uniform(0, 1, [d,])
        pdf /= np.sum(pdf)
    elif mode_pdf=="uniform":
        pdf = np.ones((d,)) / d
    else:
        #tmp = np.arange(0, d)
        #pdf = np.exp(-(tmp-d/2)**2/1000.)
        #pdf /= np.sum(pdf)

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


def meas_gen_pdf_tensor(sig, m, num_meas, pdf, sigma=0.):
    """
    Generates the set of measurements from the signal sig based on a pdf
    This function works with the tensor sig. Remember you had to separate this function and the one below due to the
    differentiation problems.
    :param sig: the signal
    :param m: the mask length
    :param dist: the distribution from which the measurements are generated
    :return: the set of measurements
    """
    meas = torch.zeros((num_meas, m))
    cumsum = np.cumsum(pdf)
    start1 = time.time()
    mat = torch.zeros((len(pdf), m))
    for j in range(len(pdf)):
        mat[j, :] = torch.cat((sig[j:], sig[0:j]), dim=0)[0:m]
    for i in range(num_meas):
        s = np.random.rand(1)
        t = int(np.sum((cumsum<s)))
        tmp = mat[t, :]
        meas[i,:] = tmp + sigma * torch.randn(tmp.shape)
    end1 = time.time() - start1

    start2 = time.time()
    for i in range(num_meas):
        s = np.random.rand(1)
        t = int(np.sum((cumsum<s)))
        tmp = torch.cat((sig[t:], sig[0:t]), dim=0)
        tmp = tmp[0:m]
        meas[i,:] = tmp + sigma * torch.randn(tmp.shape)
    end2 = time.time() - start2
    import pdb; pdb.set_trace()
    return meas


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
    #for i in range(num_meas):
    #    tmp = mat[shifts[i], :]
    #    meas[i,:] = tmp + sigma * torch.randn(tmp.shape)
    #import pdb; pdb.set_trace()
    #for i in range(num_meas):
    #    j = shifts[i]
    #    tmp = torch.cat((sig[j:], sig[0:j]), dim=0)
    #    tmp = tmp[0:m]
    #    meas[i, :] = tmp + sigma * torch.randn(tmp.shape)
    #end2 = time.time() - start2
    #import pdb; pdb.set_trace()
    return meas

def find_dist(sig, ref, exp_name):
    """
    Plots the two signals and compare them to one another
    :param sig: the estimated signal
    :param ref: the reference gt signal
    :return: none
    """
    # align the ref signal and the recovered one
    res = -1*float('inf')
    for s in range(len(sig)):
        tmp = np.concatenate((sig[s:], sig[0:s]), axis=0)
        inner_prod = np.sum(tmp*ref)
        if inner_prod>res:
            index = s
            res = inner_prod
    sig_aligned = np.concatenate((sig[index:], sig[0:index]), axis=0)
    figure = plt.figure()
    plt.plot(sig_aligned, label='predicted')
    plt.plot(ref, label='gt.')
    plt.legend()
    plt.savefig('./results/'+exp_name+'/compare.png')
    return figure


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

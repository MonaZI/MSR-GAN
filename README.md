# MSR-GAN

Implementation for the paper:

MSR-GAN: Multi-Segment Reconstruction via Adversarial Learning, accepted in ICASSP 2021.

by Mona Zehni, Zhizhen Zhao

Link to paper (arxiv):

## Pre-requisites
- Pytorch, Tensorboard, Numpy, Matplotlib

## Run
Examples of how to run the code for three different scenarios are given below:
- Known PMF
```r
python main_wpdf.py -lrate 0.005 -batch_size 200 -gumbel -num_epoch 9000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 24 -mid_size 100 -mode_sig periodic -snr 0 -num_meas 50000 -seed 1
``` 

## More information
If you find this repositry helpful in your publications, please consider citing our paper.

If you have any questions, please contact Mona Zehni (mzehni2@illinois.edu).

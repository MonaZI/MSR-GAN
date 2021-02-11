# MSR-GAN

Implementation for the paper:

MSR-GAN: Multi-Segment Reconstruction via Adversarial Learning, accepted in ICASSP 2021.

by Mona Zehni, Zhizhen Zhao

Link to paper (arxiv):

## Pre-requisites
- Pytorch, Tensorboard, Numpy, Matplotlib

## Run
Examples of how to run the code for three different scenarios are given below:
- Unknown PMF
```r
python main_wpdf.py -lrate 0.005 -batch_size 200 -gumbel -num_epoch 5000 -pdf_periodic -sig_len 64 -mask_len 24 -mid_size 100 -mode_sig tri -num_meas 50000 -seed 0
```
- Fixed PMF with uniform distribution
```r
python main_wpdf.py -batch_size 200 -gumbel -num_epoch 12000 -pdf_periodic -sig_len 64 -mask_len 24 -mid_size 100 -expName exp10_1 -mode_sig tri -num_meas 50000 -seed 0 -unif_pdf
```
- Known PMF
```r
python main_wpdf.py -lrate 0.004 -batch_size 200 -gumbel -num_epoch 12000 -pdf_periodic -sig_len 64 -mask_len 24 -mid_size 100 -mode_sig tri -num_meas 50000 -seed 0 -correct_pdf
```

## More information
If you find this repositry helpful in your publications, please consider citing our paper.

If you have any questions, please contact Mona Zehni (mzehni2@illinois.edu).

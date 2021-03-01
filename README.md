# MSR-GAN

Implementation for the paper:

MSR-GAN: Multi-Segment Reconstruction via Adversarial Learning, accepted in ICASSP 2021.

by Mona Zehni, Zhizhen Zhao

Link to paper (arxiv): https://arxiv.org/abs/2102.09494

## Pre-requisites
- Pytorch, Tensorboard, Numpy, Matplotlib

## Run
Examples of how to run the code for three different scenarios are given below:
- Unknown PMF
```r
python main_wpdf.py -lrate 0.005 -batch_size 200 -gumbel -num_epoch 10000 -pdf_periodic -sig_len 64 -mask_len 24 -mid_size 100 -mode_sig tri -num_meas 50000 -seed 0 -exp_name test_unknown
```
- Fixed PMF with uniform distribution
```r
python main_wpdf.py -batch_size 200 -gumbel -num_epoch 20000 -pdf_periodic -sig_len 64 -mask_len 24 -mid_size 100 -mode_sig tri -num_meas 50000 -seed 0 -unif_pdf -exp_name test_fixed
```
- Known PMF
```r
python main_wpdf.py -lrate 0.004 -batch_size 200 -gumbel -num_epoch 20000 -pdf_periodic -sig_len 64 -mask_len 24 -mid_size 100 -mode_sig tri -num_meas 50000 -seed 0 -correct_pdf -exp_name test_known
```

## More information
If you find this repositry helpful in your publications, please consider citing our paper.
```r
@ARTICLE{MSR-GAN,
       author = {{Zehni}, Mona and {Zhao}, Zhizhen},
        title = "{MSR-GAN: Multi-Segment Reconstruction via Adversarial Learning}",
      journal = {arXiv e-prints},
         year = 2021,
        month = feb,
        pages = {arXiv:2102.09494},
archivePrefix = {arXiv},
       eprint = {2102.09494},
 primaryClass = {eess.SP}}
```
If you have any questions, please contact Mona Zehni (mzehni2@illinois.edu).

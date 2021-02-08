# for all these experiments we have: -lrate 0.008 -lrate_x 0.002 -lrate_pdf 0.001
python main_wpdf.py -batch_size 200 -gumbel -num_epoch 12000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -expName exp0_1 -mode_sig tri -snr 0 -num_meas 50000 -seed 0 > runs/run0_1
python main_wpdf.py -batch_size 200 -gumbel -num_epoch 12000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -expName exp1_1 -mode_sig tri -snr 0 -num_meas 50000 -seed 0 -unif_pdf > runs/run1_1
python main_wpdf.py -batch_size 200 -gumbel -num_epoch 12000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -expName exp2_1 -mode_sig tri -snr 0 -num_meas 50000 -seed 0 -correct_pdf > runs/run2_1

python main_wpdf.py -batch_size 200 -gumbel -num_epoch 12000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -expName exp3_1 -mode_sig random -snr 0 -num_meas 50000 -seed 1 > runs/run3_1
python main_wpdf.py -batch_size 200 -gumbel -num_epoch 12000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -expName exp4_1 -mode_sig random -snr 0 -num_meas 50000 -seed 1 -unif_pdf > runs/run4_1
python main_wpdf.py -batch_size 200 -gumbel -num_epoch 12000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -expName exp5_1 -mode_sig random -snr 0 -num_meas 50000 -seed 1 -correct_pdf > runs/run5_1

python main_wpdf.py -batch_size 200 -gumbel -num_epoch 12000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -expName exp6_1 -mode_sig periodic -snr 0 -num_meas 50000 -seed 2 > runs/run6_1
python main_wpdf.py -batch_size 200 -gumbel -num_epoch 12000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -expName exp7_1 -mode_sig periodic -snr 0 -num_meas 50000 -seed 2 -unif_pdf > runs/run7_1
python main_wpdf.py -batch_size 200 -gumbel -num_epoch 12000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -expName exp8_1 -mode_sig periodic -snr 0 -num_meas 50000 -seed 2 -correct_pdf > runs/run8_1



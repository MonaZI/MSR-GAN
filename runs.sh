python main_wpdf.py -batch_size 200 -gumbel -num_epoch 50000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -lrate 0.003 -lrate_x 0.001 -lrate_pdf 0.0005 -expName exp0_1 -mode_sig tri -snr 1 -num_meas 50000 -seed 0 > runs/run0_1

python main_wpdf.py -batch_size 200 -gumbel -num_epoch 20000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -lrate 0.0003 -lrate_x 0.0001 -lrate_pdf 0.0001 -expName exp1_1 -mode_sig tri -snr 1 -num_meas 50000 -unif_pdf -tau 0.001 -seed 0 > runs/run1_1

python main_wpdf.py -batch_size 200 -gumbel -num_epoch 50000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -lrate 0.003 -lrate_x 0.001 -lrate_pdf 0.001 -expName exp2_1 -mode_sig tri -snr 1 -num_meas 50000 -correct_pdf -tau 0.001 -seed 0 > runs/run2_1

python main_wpdf.py -batch_size 200 -gumbel -num_epoch 50000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -lrate 0.003 -lrate_x 0.001 -lrate_pdf 0.001 -expName exp3_1 -mode_sig random -snr 1 -num_meas 50000 -seed 1 > runs/run3_1

python main_wpdf.py -batch_size 200 -gumbel -num_epoch 20000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -lrate 0.0003 -lrate_x 0.0001 -lrate_pdf 0.0001 -expName exp4_1 -mode_sig random -snr 1 -num_meas 50000 -unif_pdf -tau 0.001 -seed 1 > runs/run4_1

python main_wpdf.py -batch_size 200 -gumbel -num_epoch 50000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -lrate 0.003 -lrate_x 0.001 -lrate_pdf 0.001 -expName exp5_1 -mode_sig random -snr 1 -num_meas 50000 -correct_pdf -tau 0.001 -seed 1 > runs/run5_1

python main_wpdf.py -batch_size 200 -gumbel -num_epoch 50000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -lrate 0.0015 -lrate_x 0.0005 -lrate_pdf 0.0005 -expName exp6_1 -mode_sig periodic -snr 1 -num_meas 50000 -seed 2 > runs/run6_1

python main_wpdf.py -batch_size 200 -gumbel -num_epoch 20000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -lrate 0.0015 -lrate_x 0.0005 -lrate_pdf 0.0005 -expName exp7_1 -mode_sig periodic -snr 1 -num_meas 50000 -unif_pdf -tau 0.001 -seed 2 > runs/run7_1

python main_wpdf.py -batch_size 200 -gumbel -num_epoch 50000 -pdf_periodic -a_size 3 -sig_len 64 -mask_len 32 -mid_size 100 -lrate 0.0015 -lrate_x 0.0005 -lrate_pdf 0.0005 -expName exp8_1 -mode_sig periodic -snr 1 -num_meas 50000 -correct_pdf -tau 0.001 -seed 2 > runs/run8_1

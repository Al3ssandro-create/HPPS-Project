export CUDA_VISIBLE_DEVICES=0,1
../gmm_um 32 ../data/mydata_4g.txt ../../Part_2/results/01_out_2g_strong.txt 8 > ../../Part_2/results/profiling/GPU_2/01_out_2g_strong_time.txt
../gmm_um 32 ../data/mydata_4g.txt ../../Part_2/results/02_out_2g_strong.txt 8 > ../../Part_2/results/profiling/GPU_2/02_out_2g_strong_time.txt
../gmm_um 32 ../data/mydata_4g.txt ../../Part_2/results/03_out_2g_strong.txt 8 > ../../Part_2/results/profiling/GPU_2/03_out_2g_strong_time.txt
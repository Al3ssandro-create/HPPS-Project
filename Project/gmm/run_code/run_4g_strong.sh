export CUDA_VISIBLE_DEVICES=0,1,2,3
../gmm_um 32 ../data/mydata_4g.txt ../../Part_2/results/01_out_4g_strong.txt 8 > ../../Part_2/results/profiling/01_out_4g_strong_time.txt
../gmm_um 32 ../data/mydata_4g.txt ../../Part_2/results/02_out_4g_strong.txt 8 > ../../Part_2/results/profiling/02_out_4g_strong_time.txt
../gmm_um 32 ../data/mydata_4g.txt ../../Part_2/results/03_out_4g_strong.txt 8 > ../../Part_2/results/profiling/03_out_4g_strong_time.txt


export CUDA_VISIBLE_DEVICES=0
../gmm_um 32 ../data/mydata_1g.txt ../../Part_2/results/01_out_1g_weak.txt 8 > ../../Part_2/results/profiling/GPU_1/01_out_1g_weak_time.txt
../gmm_um 32 ../data/mydata_1g.txt ../../Part_2/results/02_out_1g_weak.txt 8 > ../../Part_2/results/profiling/GPU_1/02_out_1g_weak_time.txt
../gmm_um 32 ../data/mydata_1g.txt ../../Part_2/results/03_out_1g_weak.txt 8 > ../../Part_2/results/profiling/GPU_1/03_out_1g_weak_time.txt

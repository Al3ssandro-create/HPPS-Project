include ../shared.mk

gmm: gaussian_UM.cu gaussian_kernel.cu gaussian.h invert_matrix.h readData.cpp invert_matrix.cpp
	$(NVCC) $(NVCC_FLAGS) $(LINK_FLAG) -Xcompiler -fopenmp -lgomp gaussian_UM.cu invert_matrix.cpp readData.cpp -o gmm_UM

clean:
	rm -f gmm mydata_*.txt 


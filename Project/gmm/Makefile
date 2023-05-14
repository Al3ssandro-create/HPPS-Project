include ../shared.mk

gmm_um: gaussian_um.cu gaussian_kernel.cu gaussian.h invert_matrix.h readData.cpp invert_matrix.cpp
	$(NVCC) $(NVCC_FLAGS) $(LINK_FLAG) -Xcompiler -fopenmp -lgomp gaussian_um.cu invert_matrix.cpp readData.cpp -o gmm_um

old: gaussian.cu gaussian_kernel.cu gaussian.h invert_matrix.h readData.cpp invert_matrix.cpp
	$(NVCC) $(NVCC_FLAGS) $(LINK_FLAG) -Xcompiler -fopenmp -lgomp gaussian.cu invert_matrix.cpp readData.cpp -o gmm_old	



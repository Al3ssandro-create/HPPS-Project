include ../shared.mk

gmm_UM: gaussian_um.cu gaussian_kernel.cu gaussian.h invert_matrix.h readData.cpp invert_matrix.cpp
	$(NVCC) $(NVCC_FLAGS) $(LINK_FLAG) -Xcompiler -fopenmp -lgomp gaussian_um.cu invert_matrix.cpp readData.cpp -o gmm_um

clean:
	rm -f gmm mydata_*.txt 


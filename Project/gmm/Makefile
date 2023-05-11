include ../shared.mk

um: gaussian_um.cu gaussian_kernel.cu gaussian.h invert_matrix.h readData.cpp invert_matrix.cpp
	$(NVCC) $(NVCC_FLAGS) $(LINK_FLAG) -Xcompiler -fopenmp -lgomp gaussian_um.cu invert_matrix.cpp readData.cpp -o gmm_um

old: gaussian.cu gaussian_kernel.cu gaussian.h invert_matrix.h readData.cpp invert_matrix.cpp
	$(NVCC) $(NVCC_FLAGS) $(LINK_FLAG) -Xcompiler -fopenmp -lgomp gaussian.cu invert_matrix.cpp readData.cpp -o gmm_old

debug: gaussian_um.cu gaussian_kernel.cu gaussian.h invert_matrix.h readData.cpp invert_matrix.cpp
	$(NVCC) $(NVCC_FLAGS) $(LINK_FLAG) -Xcompiler -fopenmp -lgomp gaussian_um.cu invert_matrix.cpp readData.cpp -g -G -o gmm_debug

fuffa: gaussian_um_fuffa.cu gaussian_kernel.cu gaussian.h invert_matrix.h readData.cpp invert_matrix.cpp
	$(NVCC) $(NVCC_FLAGS) $(LINK_FLAG) -Xcompiler -fopenmp -lgomp gaussian_um_fuffa.cu invert_matrix.cpp readData.cpp -o gmm_fuffa

clean:
	rm -f gmm mydata_*.txt 


include ../shared.mk

gmm: gaussian.cu gaussian_kernel.cu gaussian.h invert_matrix.h readData.cpp invert_matrix.cpp
	$(NVCC) $(NVCC_FLAGS) $(LINK_FLAG) -Xcompiler -fopenmp -lgomp gaussian.cu invert_matrix.cpp readData.cpp -o gmm
	python data/datagen.py 4 100000 mydata_1g.txt
	python data/datagen.py 4 200000 mydata_2g.txt
	python data/datagen.py 4 400000 mydata_4g.txt
	python data/datagen.py 4 800000 mydata_8g.txt

clean:
	rm -f gmm mydata_*.txt 


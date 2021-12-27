object=model.o
NVCC=/usr/local/cuda/bin/nvcc
CC=g++
CUDAFLAGS= -std=c++11 
INCDIRS=-I /usr/local/cuda-10.2/samples/common/inc

model.o:vgg16_main.cu model.hpp utils.h layers.cuh
	$(NVCC)	$(CUDAFLAGS)	$(INCDIRS)	vgg16_main.cu	-o	model.o

.PHONY:clean
clean:
	rm	*.o
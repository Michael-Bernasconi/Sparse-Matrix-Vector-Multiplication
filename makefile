CC=gcc
NVCC=nvcc
MPICC=mpicxx

# Flags per A100/A30 del cluster
NVCCFLAGS=-O3 -arch=sm_80 -ccbin=$(MPICC) -Xcompiler -fopenmp
INCLUDES=-I./include
LIBS=-lm

TARGET=bin/cuda-SpMV-multi-gpu

SRCS=src/cuda-SpMV-CSR-multi.cu src/matrix_utils.c src/my_time_lib.c
OBJS=obj/matrix_utils.o obj/my_time_lib.o

all: $(TARGET)

$(TARGET): $(SRCS)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(TARGET) $(INCLUDES) $(LIBS)

clean:
	rm -rf bin obj
CC=gcc
NVCC=nvcc
MPICC=mpicxx

# Flags for A100/A30 cluster
# Added -lcusparse to LIBS to link the NVIDIA Sparse library
NVCCFLAGS=-O3 -arch=sm_80 -ccbin=$(MPICC) -Xcompiler -fopenmp
INCLUDES=-I./include
LIBS=-lm -lcusparse

# Updated Target name for cuSPARSE
TARGET=bin/cuda-SpMV-cusparse-multi

# Main source file switched to cuSPARSE version
SRCS=src/cuda-SpMV-cusparse-multi.cu src/matrix_utils.c src/my_time_lib.c
OBJS=obj/matrix_utils.o obj/my_time_lib.o

all: $(TARGET)

$(TARGET): $(SRCS)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(TARGET) $(INCLUDES) $(LIBS)

clean:
	rm -rf bin obj
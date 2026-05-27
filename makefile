CC=gcc
NVCC=nvcc
MPICC=mpicxx

# Flags for A100/A30 cluster
NVCCFLAGS=-O3 -arch=sm_80 -ccbin=$(MPICC) -Xcompiler -fopenmp
INCLUDES=-I./include
# For custom COO, we only need the math library. 
# Add -lcusparse if you plan to mix with cuSPARSE calls.
LIBS=-lm 

# Target name for COO
TARGET=bin/cuda-SpMV-coo-multi

# Source files for COO
SRCS=src/cuda-SpMV-COO-multi.cu src/matrix_utils.c src/my_time_lib.c
OBJS=obj/matrix_utils.o obj/my_time_lib.o

all: $(TARGET)

$(TARGET): $(SRCS)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(TARGET) $(INCLUDES) $(LIBS)

clean:
	rm -rf bin obj
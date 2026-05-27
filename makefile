CC=gcc
NVCC=nvcc
MPICC=mpicxx

NVCCFLAGS=-O3 -arch=sm_80 -ccbin=$(MPICC) -Xcompiler -fopenmp
INCLUDES=-I./include
LIBS=-lm -lcusparse

BIN_FOLDER := bin
SRC_FOLDER := src

TARGETS = $(BIN_FOLDER)/cuda-SpMV-CSR-multi \
          $(BIN_FOLDER)/cuda-SpMV-COO-multi \
          $(BIN_FOLDER)/cuda-SpMV-CSR-Vector-multi \
          $(BIN_FOLDER)/cuda-SpMV-cuSparse-multi

all: $(TARGETS)

$(BIN_FOLDER)/%-multi: $(SRC_FOLDER)/%-multi.cu $(SRC_FOLDER)/matrix_utils.c $(SRC_FOLDER)/my_time_lib.c
	@mkdir -p $(BIN_FOLDER)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

clean:
	rm -rf $(BIN_FOLDER) obj
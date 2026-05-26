# Compiler Definitions
CC=gcc
NVCC=nvcc

# C Compiler Flags: Added -g for debug symbols (required for Valgrind)
CFLAGS=-O3 -Wall -g -fopenmp

# CUDA Compiler Flags: Target set to sm_80 (Standard for Cluster A100/A30 GPUs)
NVCCFLAGS=-O3 -arch=sm_80 --use_fast_math -Xptxas -v -g -lineinfo

# Libraries and Include paths
LIBS=-lm -lgomp
CUSPARSE_LIBS=-lcusparse
INCLUDES=-I./include

# Directory Structure
BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src

# Executable Targets 
TARGETS = $(BIN_FOLDER)/cpu-SpMV-CSR \
          $(BIN_FOLDER)/cuda-SpMV-COO \
          $(BIN_FOLDER)/cuda-SpMV-CSR \
          $(BIN_FOLDER)/cuda-SpMV-CSR-Vector \
          $(BIN_FOLDER)/cuda-SpMV-cuSparse \
          $(BIN_FOLDER)/deviceQuery

# Default target
all: $(TARGETS)

# Special target for profiling (custom iterations via command line)
profile_build: CFLAGS += $(ITER)
profile_build: NVCCFLAGS += $(ITER)
profile_build: all

$(OBJ_FOLDER)/%.o: $(SRC_FOLDER)/%.c
	@mkdir -p $(OBJ_FOLDER)
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDES)

$(BIN_FOLDER)/cpu-SpMV-CSR: $(SRC_FOLDER)/cpu-SpMV-CSR.c $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/matrix_utils.o
	@mkdir -p $(BIN_FOLDER)
	$(CC) $(CFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

$(BIN_FOLDER)/cuda-SpMV-COO: $(SRC_FOLDER)/cuda-SpMV-COO.cu $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/matrix_utils.o
	@mkdir -p $(BIN_FOLDER)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

$(BIN_FOLDER)/cuda-SpMV-CSR: $(SRC_FOLDER)/cuda-SpMV-CSR.cu $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/matrix_utils.o
	@mkdir -p $(BIN_FOLDER)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

$(BIN_FOLDER)/cuda-SpMV-CSR-Vector: $(SRC_FOLDER)/cuda-SpMV-CSR-Vector.cu $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/matrix_utils.o
	@mkdir -p $(BIN_FOLDER)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

$(BIN_FOLDER)/cuda-SpMV-cuSparse: $(SRC_FOLDER)/cuda-SpMV-cuSparse.cu $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/matrix_utils.o
	@mkdir -p $(BIN_FOLDER)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(INCLUDES) $(LIBS) $(CUSPARSE_LIBS)

$(BIN_FOLDER)/deviceQuery: $(SRC_FOLDER)/deviceQuery.cpp
	@mkdir -p $(BIN_FOLDER)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)

.PHONY: all clean profile_build

# Compiler Definitions
CC=gcc
NVCC=nvcc

# C Compiler Flags: Added -g for debug symbols (required for Valgrind)
CFLAGS=-O3 -Wall -g -fopenmp

# CUDA Compiler Flags: Added -g and -lineinfo for profiling (required for Nsight Compute)
# -arch=sm_89: Targets NVIDIA Ada Lovelace architecture (RTX 40 series)
NVCCFLAGS=-O3 -arch=sm_89 --use_fast_math -Xptxas -v -g -lineinfo

# Libraries and Include paths
LIBS=-lm -lgomp
INCLUDES=-I./include

# Directory Structure
BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src

# Executable Targets
TARGETS = $(BIN_FOLDER)/cpu-SpMV-COO \
          $(BIN_FOLDER)/cpu-SpMV-CSR \
          $(BIN_FOLDER)/cuda-SpMV-COO \
          $(BIN_FOLDER)/cuda-SpMV-CSR

# Default target
all: $(TARGETS)

# Special target for profiling (overrides iterations from the command line)
# Usage: make profile_build ITER="-DBENCHMARK_ITERATIONS=1 -DWARMUP_ITERATIONS=0"
profile_build: CFLAGS += $(ITER)
profile_build: NVCCFLAGS += $(ITER)
profile_build: all

# --- Compilation of Shared Objects ---

$(OBJ_FOLDER)/my_time_lib.o: $(SRC_FOLDER)/my_time_lib.c
	@mkdir -p $(OBJ_FOLDER)
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDES)

$(OBJ_FOLDER)/matrix_utils.o: $(SRC_FOLDER)/matrix_utils.c
	@mkdir -p $(OBJ_FOLDER)
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDES)

# --- CPU Executables ---

$(BIN_FOLDER)/cpu-SpMV-COO: $(SRC_FOLDER)/cpu-SpMV-COO.c $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/matrix_utils.o
	@mkdir -p $(BIN_FOLDER)
	$(CC) $(CFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

$(BIN_FOLDER)/cpu-SpMV-CSR: $(SRC_FOLDER)/cpu-SpMV-CSR.c $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/matrix_utils.o
	@mkdir -p $(BIN_FOLDER)
	$(CC) $(CFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

# --- GPU (CUDA) Executables ---

$(BIN_FOLDER)/cuda-SpMV-COO: $(SRC_FOLDER)/cuda-SpMV-COO.cu $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/matrix_utils.o
	@mkdir -p $(BIN_FOLDER)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

$(BIN_FOLDER)/cuda-SpMV-CSR: $(SRC_FOLDER)/cuda-SpMV-CSR.cu $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/matrix_utils.o
	@mkdir -p $(BIN_FOLDER)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

# --- Utilities ---

clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)

.PHONY: all clean profile_build
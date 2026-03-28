# Compiler Definitions
CC=gcc
NVCC=nvcc

# C Compiler Flags: High optimization and all warnings
CFLAGS=-O3 -Wall

# CUDA Compiler Flags:
# -O3: Optimization level 3
# -arch=sm_89: Targets NVIDIA Ada Lovelace architecture (e.g., RTX 4090)
# --use_fast_math: Uses faster, less precise hardware-based math functions
# -Xptxas -v: Asks the PTX assembler to print verbose info (register usage, etc.)
NVCCFLAGS=-O3 -arch=sm_89 --use_fast_math -Xptxas -v

# Libraries and Include paths
LIBS=-lm
INCLUDES=-I./include

# Directory Structure
BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src

# List of final executable targets
TARGETS = $(BIN_FOLDER)/cpu-SpMV-COO \
          $(BIN_FOLDER)/cpu-SpMV-CSR \
          $(BIN_FOLDER)/cuda-SpMV-COO \
          $(BIN_FOLDER)/cuda-SpMV-CSR

# Default target: builds everything
all: $(TARGETS)

# --- Compilation of Shared Objects ---

# Compiles the timing and statistics library
$(OBJ_FOLDER)/my_time_lib.o: $(SRC_FOLDER)/my_time_lib.c
	@mkdir -p $(OBJ_FOLDER)
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDES)

# Compiles matrix loading and utility functions
$(OBJ_FOLDER)/matrix_utils.o: $(SRC_FOLDER)/matrix_utils.c
	@mkdir -p $(OBJ_FOLDER)
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDES)

# --- CPU Executables Targets ---

# Linker for CPU COO implementation
$(BIN_FOLDER)/cpu-SpMV-COO: $(SRC_FOLDER)/cpu-SpMV-COO.c $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/matrix_utils.o
	@mkdir -p $(BIN_FOLDER)
	$(CC) $(CFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

# Linker for CPU CSR implementation
$(BIN_FOLDER)/cpu-SpMV-CSR: $(SRC_FOLDER)/cpu-SpMV-CSR.c $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/matrix_utils.o
	@mkdir -p $(BIN_FOLDER)
	$(CC) $(CFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

# --- GPU (CUDA) Executables Targets ---

# Linker/Compiler for GPU COO implementation
$(BIN_FOLDER)/cuda-SpMV-COO: $(SRC_FOLDER)/cuda-SpMV-COO.cu $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/matrix_utils.o
	@mkdir -p $(BIN_FOLDER)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

# Linker/Compiler for GPU CSR implementation
$(BIN_FOLDER)/cuda-SpMV-CSR: $(SRC_FOLDER)/cuda-SpMV-CSR.cu $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/matrix_utils.o
	@mkdir -p $(BIN_FOLDER)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

# --- Utility Commands ---

# Removes all compiled binaries and object files
clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)

# Marks 'all' and 'clean' as phony targets (they aren't actual files)
.PHONY: all clean
CC=gcc
CXX=g++
NVCC=nvcc
MPICC=mpicxx

# Flags unified for performance and Ampere architecture (A30 cluster node)
NVCCFLAGS=-O3 -arch=sm_80 -ccbin=$(MPICC) -Xcompiler -fopenmp
CXXFLAGS=-O3 -std=c++14 -fopenmp

# Include and cross-library paths
INCLUDES=-I./include -I./baselinemultigpu/include
LIBS=-lm -lcusparse -lcudart -lcuda

BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
BASE_FOLDER := baselinemultigpu

# Professors baseline objects
PROF_OBJS = $(OBJ_FOLDER)/mmio.o \
            $(OBJ_FOLDER)/matrix_parser.o \
            $(OBJ_FOLDER)/import_sparse_matrix.o \
            $(OBJ_FOLDER)/utils.o

# Final executables target list
TARGETS = $(BIN_FOLDER)/cuda-SpMV-CSR-multi \
          $(BIN_FOLDER)/cuda-SpMV-COO-multi \
          $(BIN_FOLDER)/cuda-SpMV-CSR-Vector-multi \
          $(BIN_FOLDER)/cuda-SpMV-cuSparse-multi \
          $(BIN_FOLDER)/prof-SpMV-baseline

all: $(TARGETS)

# Rule for building your custom Multi-GPU CUDA kernels
$(BIN_FOLDER)/%-multi: $(SRC_FOLDER)/%-multi.cu $(SRC_FOLDER)/matrix_utils.c $(SRC_FOLDER)/my_time_lib.c
	@mkdir -p $(BIN_FOLDER)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

# Rule for compiling the professors mmio utility using CXX to handle <cstdio>
$(OBJ_FOLDER)/mmio.o: $(BASE_FOLDER)/include/mmio.c
	@mkdir -p $(OBJ_FOLDER)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Pattern rule for compiling professors cpp dependencies
$(OBJ_FOLDER)/%.o: $(BASE_FOLDER)/include/%.cpp
	@mkdir -p $(OBJ_FOLDER)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDES)

# Final link step for professors SpMV baseline
# CRITICAL FIX: Explicitly excluded $(OBJ_FOLDER)/utils.o from this link step to avoid multiple definition errors with matrix_parser.o
$(BIN_FOLDER)/prof-SpMV-baseline: $(BASE_FOLDER)/SpMV.cpp $(OBJ_FOLDER)/mmio.o $(OBJ_FOLDER)/matrix_parser.o $(OBJ_FOLDER)/import_sparse_matrix.o
	@mkdir -p $(BIN_FOLDER)
	$(MPICC) $(CXXFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)
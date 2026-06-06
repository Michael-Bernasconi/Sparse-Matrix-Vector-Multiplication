CC=gcc
CXX=g++
NVCC=nvcc
MPICC=mpicxx

NVCCFLAGS=-O3 -arch=sm_80 -ccbin=$(MPICC) -Xcompiler -fopenmp
CXXFLAGS=-O3 -std=c++14 -fopenmp

INCLUDES=-I./include -I./baselinemultigpu/include
LIBS=-lm -lcusparse -lcudart -lcuda

BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
BASE_FOLDER := baselinemultigpu

PROF_OBJS = $(OBJ_FOLDER)/mmio.o \
            $(OBJ_FOLDER)/matrix_parser.o \
            $(OBJ_FOLDER)/import_sparse_matrix.o \
            $(OBJ_FOLDER)/utils.o

TARGETS = $(BIN_FOLDER)/cuda-SpMV-CSR-multi \
          $(BIN_FOLDER)/cuda-SpMV-COO-multi \
          $(BIN_FOLDER)/cuda-SpMV-CSR-Vector-multi \
          $(BIN_FOLDER)/cuda-SpMV-cuSparse-multi \
          $(BIN_FOLDER)/prof-SpMV-baseline

all: $(TARGETS)

$(BIN_FOLDER)/%-multi: $(SRC_FOLDER)/%-multi.cu $(SRC_FOLDER)/matrix_utils.c $(SRC_FOLDER)/my_time_lib.c
	@mkdir -p $(BIN_FOLDER)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

$(OBJ_FOLDER)/mmio.o: $(BASE_FOLDER)/include/mmio.c
	@mkdir -p $(OBJ_FOLDER)
	$(CC) -O2 -c $< -o $@

$(OBJ_FOLDER)/%.o: $(BASE_FOLDER)/include/%.cpp
	@mkdir -p $(OBJ_FOLDER)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDES)

$(BIN_FOLDER)/prof-SpMV-baseline: $(BASE_FOLDER)/SpMV.cpp $(PROF_OBJS)
	@mkdir -p $(BIN_FOLDER)
	$(MPICC) $(CXXFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)
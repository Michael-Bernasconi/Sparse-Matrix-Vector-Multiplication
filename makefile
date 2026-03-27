CC=gcc
NVCC=nvcc
LIBS=-lm
INCLUDES=-I./include

# Cartelle
BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src

all: $(BIN_FOLDER)/sequentialSpMV $(BIN_FOLDER)/cudaSpMV

# Compiliamo gli oggetti comuni (Libreria tempo e Utils Matrice)
$(OBJ_FOLDER)/my_time_lib.o: $(SRC_FOLDER)/my_time_lib.c
	@mkdir -p $(OBJ_FOLDER)
	$(CC) -c $< -o $@ $(INCLUDES)

$(OBJ_FOLDER)/matrix_utils.o: $(SRC_FOLDER)/matrix_utils.c
	@mkdir -p $(OBJ_FOLDER)
	$(CC) -c $< -o $@ $(INCLUDES)

# Target CPU: usa sia time_lib che matrix_utils
$(BIN_FOLDER)/sequentialSpMV: $(SRC_FOLDER)/sequentialSpMV.c $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/matrix_utils.o
	@mkdir -p $(BIN_FOLDER)
	$(CC) $^ -o $@ $(INCLUDES) $(LIBS)

# Target GPU: DEVE includere matrix_utils.o per trovare load_matrix_market_to_csr
$(BIN_FOLDER)/cudaSpMV: $(SRC_FOLDER)/cudaSpMV.cu $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/matrix_utils.o
	@mkdir -p $(BIN_FOLDER)
	$(NVCC) $^ -o $@ $(INCLUDES) $(LIBS)

clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)
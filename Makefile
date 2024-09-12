# Compiler and flags
CXX = g++
CXXFLAGS = -I./src -I./third-party/include -std=c++17 -w -DLLVM_VERSION_STRING="\"<LLVM_VERSION>\""
LDFLAGS = -ldl -L./third-party/lib -Wl,-rpath=./third-party/lib -lxla_extension  # Add third-party libraries here

# Directories
SRC_DIR = ./src
BUILD_DIR = ./build
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin

# Project name
TARGET = $(BIN_DIR)/app

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# Default target
all: $(TARGET)

# Rule to link object files and create the executable
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) $(OBJS) $(LDFLAGS) -o $(TARGET)

# Rule to compile each .cpp file into an object file
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create the binary directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Create the object directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean the build
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all clean

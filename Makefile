# Makefile - builds the Parallel-Computing-Project
# Usage:
#   make         # build release executable (parallel_app)
#   make debug   # build with debug symbols
#   make clean   # remove objects, deps and executable
#   make run     # run the produced executable
#
# Notes:
# - Uses g++ by default. If you want to use another compiler set CXX on the make command line,
#   e.g. `make CXX=clang++`.
# - This Makefile uses -MMD -MP to generate header dependency files (.d). It is compatible with
#   typical Unix-like make environments (MSYS/MinGW, WSL, Linux, macOS). On plain Windows + nmake/MSVC,
#   you'll need a different project file or use MSYS/Mingw make.
# End of header comments

CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -I. -Imatmul_algorithms -Itest
LDFLAGS :=
LDLIBS :=

# sources (root, matmul_algorithms, test)
SRCS := $(wildcard *.cpp) $(wildcard matmul_algorithms/*.cpp) $(wildcard test/*.cpp)
OBJS := $(SRCS:.cpp=.o)
DEPS := $(SRCS:.cpp=.d)

TARGET := parallel_app

.PHONY: all clean run debug help

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# compile with dependency generation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# include generated dependency files (no error if they don't exist yet)
-include $(DEPS)

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET) $(STRASSEN_OMP_TEST_TARGET) $(STRASSEN_MPI_TEST_TARGET)

run: $(TARGET)
	./$(TARGET)

debug: CXXFLAGS += -g -O0
debug: clean all

help:
	@echo "Makefile targets:"
	@echo "  make        - build release executable ($(TARGET))"
	@echo "  make debug  - build with debug symbols"
	@echo "  make clean  - remove objects, deps and executable"
	@echo "  make run    - run the produced executable"
	@echo "  make strassen_test - build and run Strassen OpenMP test"
	@echo "  make strassen_test_mpi - build and run Strassen MPI test with $(NUM_PROCESSES) processes"
	@echo "Notes: set CXX to change compiler, e.g. 'make CXX=clang++'"

# Strassen omp test target
NUM_THREADS := 4
STRASSEN_OMP_TEST_TARGET := strassen_test_app_omp
STRASSEN_OMP_TEST_SRC := test/test_strassen/test_omp.cpp

strassen_test: $(STRASSEN_OMP_TEST_TARGET)
	OMP_NUM_THREADS=$(NUM_THREADS) ./$(STRASSEN_OMP_TEST_TARGET)

$(STRASSEN_OMP_TEST_TARGET): $(STRASSEN_OMP_TEST_SRC)
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $^

# Strassen MPI test target
NUM_PROCESSES := 4
STRASSEN_MPI_TEST_TARGET := strassen_test_app_mpi
STRASSEN_MPI_TEST_SRC := test/test_strassen/test_mpi.cpp
MPICXX := mpic++

strassen_test_mpi: $(STRASSEN_MPI_TEST_TARGET)
	mpirun -np $(NUM_PROCESSES) ./$(STRASSEN_MPI_TEST_TARGET)

$(STRASSEN_MPI_TEST_TARGET): $(STRASSEN_MPI_TEST_SRC) utils.cpp
	$(MPICXX) $(CXXFLAGS) -DOMPI_SKIP_MPICXX -o $@ $^
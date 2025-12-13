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

CXX := mpic++
CXXFLAGS := -std=c++23 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wswitch-enum -Wformat=2 -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual -fopenmp -I. -Imatmul_algorithms -Itest -Istrassen_utils
LDFLAGS := -fopenmp
LDLIBS := -lmpi

# sources (root, matmul_algorithms, test)
SRCS := $(wildcard *.cpp) $(wildcard matmul_algorithms/*.cpp) $(wildcard test/*.cpp) $(wildcard strassen_utils/*.cpp) $(wildcard naive_utils/*.cpp)
OBJS := $(SRCS:.cpp=.o)
DEPS := $(SRCS:.cpp=.d)

TARGET := parallel_app

.PHONY: all clean run mpirun debug help strassen_test hybrid_test omp_test mpi_test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# compile with dependency generation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# include generated dependency files (no error if they don't exist yet)
-include $(DEPS)

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET) $(STRASSEN_OMP_TEST_TARGET) $(STRASSEN_MPI_TEST_TARGET) $(STRASSEN_HYBRID_TEST_TARGET) $(NAIVETEST_OMP_TEST_TARGET) $(NAIVETEST_MPI_TEST_TARGET) $(NAIVETEST_HYBRID_TEST_TARGET)

run: $(TARGET)
	./$(TARGET)

# MPI run target
NP := 4
mpirun: $(TARGET)
	mpirun -np $(NP) ./$(TARGET)

debug: CXXFLAGS += -g -O0
debug: clean all

help:
	@echo "Makefile targets:"
	@echo "  make        - build release executable ($(TARGET)) with MPI+OpenMP support"
	@echo "  make debug  - build with debug symbols"
	@echo "  make clean  - remove objects, deps and executable"
	@echo "  make run    - run the produced executable"
	@echo "  make mpirun - run with MPI (4 processes by default, set NP=n to change)"
	@echo "  make omp_test - build and run Strassen OpenMP tests"
	@echo "  make hybrid_test - build and run Strassen Hybrid (MPI+OpenMP) test"
	@echo "  make mpi_test - build and run Strassen MPI test"
	@echo "Notes: Uses mpic++ compiler with OpenMP and MPI support"

# Strassen OpenMP test target
STRASSEN_OMP_TEST_TARGET := strassen_test_omp
STRASSEN_OMP_TEST_SRC := test/test_strassen/test_omp.cpp

omp_test: $(STRASSEN_OMP_TEST_TARGET)
	./$(STRASSEN_OMP_TEST_TARGET)

$(STRASSEN_OMP_TEST_TARGET): $(STRASSEN_OMP_TEST_SRC) $(filter-out main.o, $(OBJS))
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Strassen hybrid test target
STRASSEN_HYBRID_TEST_TARGET := strassen_test_app_hybrid
STRASSEN_HYBRID_TEST_SRC := test/test_strassen/test_hybrid.cpp
HYBRID_NP := 4

hybrid_test: $(STRASSEN_HYBRID_TEST_TARGET)
	OMP_NUM_THREADS=$(NUM_THREADS) mpirun -np $(HYBRID_NP) ./$(STRASSEN_HYBRID_TEST_TARGET)

$(STRASSEN_HYBRID_TEST_TARGET): $(STRASSEN_HYBRID_TEST_SRC) $(filter-out main.o, $(OBJS))
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

# Strassen MPI test target
STRASSEN_MPI_TEST_TARGET := strassen_test_app_mpi
STRASSEN_MPI_TEST_SRC := test/test_strassen/test_mpi.cpp
MPI_NP := 4

mpi_test: $(STRASSEN_MPI_TEST_TARGET)
	mpirun -np $(MPI_NP) ./$(STRASSEN_MPI_TEST_TARGET)

$(STRASSEN_MPI_TEST_TARGET): $(STRASSEN_MPI_TEST_SRC) $(filter-out main.o, $(OBJS))
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

NAIVETEST_OMP_TEST_TARGET := naive_test_omp_api
NAIVETEST_OMP_TEST_SRC    := test/test_naive/naive_omp.cpp
NP := 1

naive_test_omp: $(NAIVETEST_OMP_TEST_TARGET)
	./$(NAIVETEST_OMP_TEST_TARGET)

$(NAIVETEST_OMP_TEST_TARGET): $(NAIVETEST_OMP_TEST_SRC) $(filter-out main.o, $(OBJS))
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)


NAIVETEST_MPI_TEST_TARGET := naive_test_mpi_api
NAIVETEST_MPI_TEST_SRC    := test/test_naive/naive_mpi.cpp

naive_test_mpi: $(NAIVETEST_MPI_TEST_TARGET)
	mpirun -np $(NP) ./$(NAIVETEST_MPI_TEST_TARGET)

$(NAIVETEST_MPI_TEST_TARGET): $(NAIVETEST_MPI_TEST_SRC) $(filter-out main.o, $(OBJS))
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

NAIVETEST_HYBRID_TEST_TARGET := naive_hybrid_test_api
NAIVETEST_HYBRID_TEST_SRC    := test/test_naive/naive_hybrid.cpp
NUM_THREADS := 4

naive_hybrid_test: $(NAIVETEST_HYBRID_TEST_TARGET)
	OMP_NUM_THREADS=$(NUM_THREADS) mpirun -np $(HYBRID_NP) ./naive_hybrid_test_api

$(NAIVETEST_HYBRID_TEST_TARGET): $(NAIVETEST_HYBRID_TEST_SRC) $(filter-out main.o, $(OBJS))
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)
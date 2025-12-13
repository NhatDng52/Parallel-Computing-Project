@echo off
echo Building Parallel Computing Project with CUDA and MSVC...
echo.

REM Set up Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM Clean previous builds
if exist *.obj del *.obj
if exist matmul_algorithms\*.obj del matmul_algorithms\*.obj
if exist test\*.obj del test\*.obj
if exist strassen_utils\*.obj del strassen_utils\*.obj
if exist parallel_app.exe del parallel_app.exe

echo.
echo Compiling CUDA files...
nvcc -allow-unsupported-compiler -O2 -I. -Imatmul_algorithms -Itest -Istrassen_utils -c matmul_algorithms\matmul_naive_gpu.cu -o matmul_algorithms\matmul_naive_gpu.obj
if errorlevel 1 (
    echo CUDA compilation failed!
    exit /b 1
)

echo Compiling C++ files with MSVC...
cl /c /O2 /EHsc /std:c++17 /I. /Imatmul_algorithms /Itest /Istrassen_utils main.cpp /Fo:main.obj
cl /c /O2 /EHsc /std:c++17 /I. /Imatmul_algorithms /Itest /Istrassen_utils utils.cpp /Fo:utils.obj
cl /c /O2 /EHsc /std:c++17 /I. /Imatmul_algorithms /Itest /Istrassen_utils matmul_algorithms\matmul_naive.cpp /Fo:matmul_algorithms\matmul_naive.obj
cl /c /O2 /EHsc /std:c++17 /I. /Imatmul_algorithms /Itest /Istrassen_utils test\correctness_test.cpp /Fo:test\correctness_test.obj
cl /c /O2 /EHsc /std:c++17 /I. /Imatmul_algorithms /Itest /Istrassen_utils test\performance_test.cpp /Fo:test\performance_test.obj

echo.
echo Linking...
nvcc -o parallel_app.exe main.obj utils.obj matmul_algorithms\matmul_naive.obj matmul_algorithms\matmul_naive_gpu.obj test\correctness_test.obj test\performance_test.obj cudart.lib

if errorlevel 1 (
    echo Linking failed!
    exit /b 1
)

echo.
echo Build successful! Executable: parallel_app.exe
echo.
echo To run: parallel_app.exe

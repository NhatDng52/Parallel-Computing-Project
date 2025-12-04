@echo off
echo Cleaning build artifacts...

if exist *.obj del *.obj
if exist matmul_algorithms\*.obj del matmul_algorithms\*.obj
if exist test\*.obj del test\*.obj
if exist strassen_utils\*.obj del strassen_utils\*.obj
if exist parallel_app.exe del parallel_app.exe
if exist *.pdb del *.pdb

echo Clean complete!

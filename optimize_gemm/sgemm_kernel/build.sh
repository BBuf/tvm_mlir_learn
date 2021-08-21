as -o sgemm_kernel_x64_fma_asm.o sgemm_kernel_x64_fma.S
gcc -O3 -mfma -o sgemm_kernel_x64_fma_its.o -c sgemm_kernel_x64_fma.c
gcc -O3 -c main.c
gcc -O3 -pthread -o sk_asm sgemm_kernel_x64_fma_asm.o main.o
gcc -O3 -pthread -o sk_its sgemm_kernel_x64_fma_its.o main.o

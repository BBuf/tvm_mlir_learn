gcc -pthread -O3 -c smtl.c
as -o cpufp_kernel_x86_avx512f.o cpufp_kernel_x86_avx512f.s
as -o cpufp_kernel_x86_avx512_vnni.o cpufp_kernel_x86_avx512_vnni.s
as -o cpufp_kernel_x86_fma.o cpufp_kernel_x86_fma.s
as -o cpufp_kernel_x86_avx.o cpufp_kernel_x86_avx.s
as -o cpufp_kernel_x86_sse.o cpufp_kernel_x86_sse.s
gcc -pthread -D_AVX512F_ -D_AVX512_VNNI_ -D_FMA_ -D_AVX_ -D_SSE_ -c cpufp_x86.c
gcc -pthread -lrt -o cpufp smtl.o cpufp_x86.o cpufp_kernel_x86_avx512f.o cpufp_kernel_x86_avx512_vnni.o cpufp_kernel_x86_fma.o cpufp_kernel_x86_avx.o cpufp_kernel_x86_sse.o

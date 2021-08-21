#ifndef _CPUFP_X86_H
#define _CPUFP_X86_H

void cpufp_kernel_x86_sse_fp32();
void cpufp_kernel_x86_sse_fp64();

void cpufp_kernel_x86_avx_fp32();
void cpufp_kernel_x86_avx_fp64();

void cpufp_kernel_x86_fma_fp32();
void cpufp_kernel_x86_fma_fp64();

void cpufp_kernel_x86_avx512f_fp32();
void cpufp_kernel_x86_avx512f_fp64();

void cpufp_kernel_x86_avx512_vnni_8b();
void cpufp_kernel_x86_avx512_vnni_8b();

#endif


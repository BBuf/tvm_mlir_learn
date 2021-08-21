.globl cpufp_kernel_x86_avx_fp32
.globl cpufp_kernel_x86_avx_fp64

cpufp_kernel_x86_avx_fp32:
    mov $0x40000000, %rax
    vxorps %ymm0, %ymm0, %ymm0
    vxorps %ymm1, %ymm1, %ymm1
    vxorps %ymm2, %ymm2, %ymm2
    vxorps %ymm3, %ymm3, %ymm3
    vxorps %ymm4, %ymm4, %ymm4
    vxorps %ymm5, %ymm5, %ymm5
    vxorps %ymm6, %ymm6, %ymm6
    vxorps %ymm7, %ymm7, %ymm7
    vxorps %ymm8, %ymm8, %ymm8
    vxorps %ymm9, %ymm9, %ymm9
    vxorps %ymm10, %ymm10, %ymm10
    vxorps %ymm11, %ymm11, %ymm11
    vxorps %ymm12, %ymm12, %ymm12
.cpufp.x86.avx.fp32.L1:
    vmulps %ymm12, %ymm12, %ymm0
    vaddps %ymm12, %ymm12, %ymm1
    vmulps %ymm12, %ymm12, %ymm2
    vaddps %ymm12, %ymm12, %ymm3
    vmulps %ymm12, %ymm12, %ymm4
    vaddps %ymm12, %ymm12, %ymm5
    vmulps %ymm12, %ymm12, %ymm6
    vaddps %ymm12, %ymm12, %ymm7
    vmulps %ymm12, %ymm12, %ymm8
    vaddps %ymm12, %ymm12, %ymm9
    vmulps %ymm12, %ymm12, %ymm10
    vaddps %ymm12, %ymm12, %ymm11
    sub $0x1, %rax
    jne .cpufp.x86.avx.fp32.L1
    ret

cpufp_kernel_x86_avx_fp64:
    mov $0x40000000, %rax
    vxorpd %ymm0, %ymm0, %ymm0
    vxorpd %ymm1, %ymm1, %ymm1
    vxorpd %ymm2, %ymm2, %ymm2
    vxorpd %ymm3, %ymm3, %ymm3
    vxorpd %ymm4, %ymm4, %ymm4
    vxorpd %ymm5, %ymm5, %ymm5
    vxorpd %ymm6, %ymm6, %ymm6
    vxorpd %ymm7, %ymm7, %ymm7
    vxorpd %ymm8, %ymm8, %ymm8
    vxorpd %ymm9, %ymm9, %ymm9
    vxorpd %ymm10, %ymm10, %ymm10
    vxorpd %ymm11, %ymm11, %ymm11
    vxorpd %ymm12, %ymm12, %ymm12
.cpufp.x86.avx.fp64.L1:
    vmulpd %ymm12, %ymm12, %ymm0
    vaddpd %ymm12, %ymm12, %ymm1
    vmulpd %ymm12, %ymm12, %ymm2
    vaddpd %ymm12, %ymm12, %ymm3
    vmulpd %ymm12, %ymm12, %ymm4
    vaddpd %ymm12, %ymm12, %ymm5
    vmulpd %ymm12, %ymm12, %ymm6
    vaddpd %ymm12, %ymm12, %ymm7
    vmulpd %ymm12, %ymm12, %ymm8
    vaddpd %ymm12, %ymm12, %ymm9
    vmulpd %ymm12, %ymm12, %ymm10
    vaddpd %ymm12, %ymm12, %ymm11
    sub $0x1, %rax
    jne .cpufp.x86.avx.fp64.L1
    ret


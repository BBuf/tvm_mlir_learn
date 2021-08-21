.globl cpufp_kernel_x86_sse_fp32
.globl cpufp_kernel_x86_sse_fp64

cpufp_kernel_x86_sse_fp32:
    mov $0x30000000, %rax
    xorps %xmm0, %xmm0
    xorps %xmm1, %xmm1
    xorps %xmm2, %xmm2
    xorps %xmm3, %xmm3
    xorps %xmm4, %xmm4
    xorps %xmm5, %xmm5
    xorps %xmm6, %xmm6
    xorps %xmm7, %xmm7
    xorps %xmm8, %xmm8
    xorps %xmm9, %xmm9
    xorps %xmm10, %xmm10
    xorps %xmm11, %xmm11
    xorps %xmm12, %xmm12
    xorps %xmm13, %xmm13
    xorps %xmm14, %xmm14
    xorps %xmm15, %xmm15
.cpufp.x86.sse.fp32.L1:
    mulps %xmm0, %xmm0
    addps %xmm1, %xmm1
    mulps %xmm2, %xmm2
    addps %xmm3, %xmm3
    mulps %xmm4, %xmm4
    addps %xmm5, %xmm5
    mulps %xmm6, %xmm6
    addps %xmm7, %xmm7
    sub $0x1, %rax
    mulps %xmm8, %xmm8
    addps %xmm9, %xmm9
    mulps %xmm10, %xmm10
    addps %xmm11, %xmm11
    mulps %xmm12, %xmm12
    addps %xmm13, %xmm13
    mulps %xmm14, %xmm14
    addps %xmm15, %xmm15
    jne .cpufp.x86.sse.fp32.L1
    ret

cpufp_kernel_x86_sse_fp64:
    mov $0x30000000, %rax
    xorpd %xmm0, %xmm0
    xorpd %xmm1, %xmm1
    xorpd %xmm2, %xmm2
    xorpd %xmm3, %xmm3
    xorpd %xmm4, %xmm4
    xorpd %xmm5, %xmm5
    xorpd %xmm6, %xmm6
    xorpd %xmm7, %xmm7
    xorpd %xmm8, %xmm8
    xorpd %xmm9, %xmm9
    xorpd %xmm10, %xmm10
    xorpd %xmm11, %xmm11
    xorpd %xmm12, %xmm12
    xorpd %xmm13, %xmm13
    xorpd %xmm14, %xmm14
    xorpd %xmm15, %xmm15
.cpufp.x86.sse.fp64.L1:
    mulpd %xmm0, %xmm0
    addpd %xmm1, %xmm1
    mulpd %xmm2, %xmm2
    addpd %xmm3, %xmm3
    mulpd %xmm4, %xmm4
    addpd %xmm5, %xmm5
    mulpd %xmm6, %xmm6
    addpd %xmm7, %xmm7
    sub $0x1, %rax
    mulpd %xmm8, %xmm8
    addpd %xmm9, %xmm9
    mulpd %xmm10, %xmm10
    addpd %xmm11, %xmm11
    mulpd %xmm12, %xmm12
    addpd %xmm13, %xmm13
    mulpd %xmm14, %xmm14
    addpd %xmm15, %xmm15
    jne .cpufp.x86.sse.fp64.L1
    ret


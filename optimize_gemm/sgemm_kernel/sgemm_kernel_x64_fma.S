.macro SK_X64_AVX_M4N24_LD
    vmovaps   0(%rdx), %ymm0
    vmovaps  32(%rdx), %ymm1
    vmovaps  64(%rdx), %ymm2
    vmovaps  96(%rdx), %ymm3
    vmovaps 128(%rdx), %ymm4
    vmovaps 160(%rdx), %ymm5
    vmovaps 192(%rdx), %ymm6
    vmovaps 224(%rdx), %ymm7
    vmovaps 256(%rdx), %ymm8
    vmovaps 288(%rdx), %ymm9
    vmovaps 320(%rdx), %ymm10
    vmovaps 352(%rdx), %ymm11
.endm

.macro SK_X64_FMA_M4N24_CP
    vmovaps  0(%rbx), %ymm12
    vmovaps 32(%rbx), %ymm13
    vmovaps 64(%rbx), %ymm14
    vbroadcastss 0(%r12, %rax, 4), %ymm15
    vfmadd231ps %ymm12, %ymm15, %ymm0
    vfmadd231ps %ymm13, %ymm15, %ymm1
    vfmadd231ps %ymm14, %ymm15, %ymm2
    vbroadcastss 0(%r13, %rax, 4), %ymm15
    vfmadd231ps %ymm12, %ymm15, %ymm3
    vfmadd231ps %ymm13, %ymm15, %ymm4
    vfmadd231ps %ymm14, %ymm15, %ymm5
    vbroadcastss 0(%r14, %rax, 4), %ymm15
    vfmadd231ps %ymm12, %ymm15, %ymm6
    vfmadd231ps %ymm13, %ymm15, %ymm7
    vfmadd231ps %ymm14, %ymm15, %ymm8
    vbroadcastss 0(%r15, %rax, 4), %ymm15
    vfmadd231ps %ymm12, %ymm15, %ymm9
    vfmadd231ps %ymm13, %ymm15, %ymm10
    vfmadd231ps %ymm14, %ymm15, %ymm11
    vmovaps  96(%rbx), %ymm12
    vmovaps 128(%rbx), %ymm13
    vmovaps 160(%rbx), %ymm14
    vbroadcastss 4(%r12, %rax, 4), %ymm15
    vfmadd231ps %ymm12, %ymm15, %ymm0
    vfmadd231ps %ymm13, %ymm15, %ymm1
    vfmadd231ps %ymm14, %ymm15, %ymm2
    vbroadcastss 4(%r13, %rax, 4), %ymm15
    vfmadd231ps %ymm12, %ymm15, %ymm3
    vfmadd231ps %ymm13, %ymm15, %ymm4
    vfmadd231ps %ymm14, %ymm15, %ymm5
    vbroadcastss 4(%r14, %rax, 4), %ymm15
    vfmadd231ps %ymm12, %ymm15, %ymm6
    vfmadd231ps %ymm13, %ymm15, %ymm7
    vfmadd231ps %ymm14, %ymm15, %ymm8
    vbroadcastss 4(%r15, %rax, 4), %ymm15
    vfmadd231ps %ymm12, %ymm15, %ymm9
    vfmadd231ps %ymm13, %ymm15, %ymm10
    vfmadd231ps %ymm14, %ymm15, %ymm11
    vmovaps 192(%rbx), %ymm12
    vmovaps 224(%rbx), %ymm13
    vmovaps 256(%rbx), %ymm14
    vbroadcastss 8(%r12, %rax, 4), %ymm15
    vfmadd231ps %ymm12, %ymm15, %ymm0
    vfmadd231ps %ymm13, %ymm15, %ymm1
    vfmadd231ps %ymm14, %ymm15, %ymm2
    vbroadcastss 8(%r13, %rax, 4), %ymm15
    vfmadd231ps %ymm12, %ymm15, %ymm3
    vfmadd231ps %ymm13, %ymm15, %ymm4
    vfmadd231ps %ymm14, %ymm15, %ymm5
    vbroadcastss 8(%r14, %rax, 4), %ymm15
    vfmadd231ps %ymm12, %ymm15, %ymm6
    vfmadd231ps %ymm13, %ymm15, %ymm7
    vfmadd231ps %ymm14, %ymm15, %ymm8
    vbroadcastss 8(%r15, %rax, 4), %ymm15
    vfmadd231ps %ymm12, %ymm15, %ymm9
    vfmadd231ps %ymm13, %ymm15, %ymm10
    vfmadd231ps %ymm14, %ymm15, %ymm11
    vmovaps 288(%rbx), %ymm12
    vmovaps 320(%rbx), %ymm13
    vmovaps 352(%rbx), %ymm14
    vbroadcastss 12(%r12, %rax, 4), %ymm15
    vfmadd231ps %ymm12, %ymm15, %ymm0
    vfmadd231ps %ymm13, %ymm15, %ymm1
    vfmadd231ps %ymm14, %ymm15, %ymm2
    vbroadcastss 12(%r13, %rax, 4), %ymm15
    vfmadd231ps %ymm12, %ymm15, %ymm3
    vfmadd231ps %ymm13, %ymm15, %ymm4
    vfmadd231ps %ymm14, %ymm15, %ymm5
    vbroadcastss 12(%r14, %rax, 4), %ymm15
    vfmadd231ps %ymm12, %ymm15, %ymm6
    vfmadd231ps %ymm13, %ymm15, %ymm7
    vfmadd231ps %ymm14, %ymm15, %ymm8
    vbroadcastss 12(%r15, %rax, 4), %ymm15
    vfmadd231ps %ymm12, %ymm15, %ymm9
    vfmadd231ps %ymm13, %ymm15, %ymm10
    vfmadd231ps %ymm14, %ymm15, %ymm11
.endm

.macro SK_X64_AVX_M4N24_ST
    vmovaps %ymm0, 0(%rdx)
    vmovaps %ymm1, 32(%rdx)
    vmovaps %ymm2, 64(%rdx)
    vmovaps %ymm3, 96(%rdx)
    vmovaps %ymm4, 128(%rdx)
    vmovaps %ymm5, 160(%rdx)
    vmovaps %ymm6, 192(%rdx)
    vmovaps %ymm7, 224(%rdx)
    vmovaps %ymm8, 256(%rdx)
    vmovaps %ymm9, 288(%rdx)
    vmovaps %ymm10, 320(%rdx)
    vmovaps %ymm11, 352(%rdx)
.endm

/* param rdi:      mat a */
/* param rsi:      mat b */
/* param rdx:      mat c */
/* param ecx:      m     */
/* param r8d:      k     */
.globl sgemm_kernel_x64_fma_m4n24
sgemm_kernel_x64_fma_m4n24:
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    movslq %r8d, %r8             /* r8: k */
    movslq %ecx, %rcx            /* rcx: m */
.SK.X64.FMA.M4N24.L1:
    xorq %rax, %rax              /* rax: index for loop a */
    movq %rsi, %rbx              /* rbx: b */
    SK_X64_AVX_M4N24_LD          /* load c to SIMD regs */
    leaq (%rdi), %r12            /* r12: a + 0 * k */ 
    leaq (%r12, %r8, 4), %r13    /* r13: a + 1 * k */
    leaq (%r13, %r8, 4), %r14    /* r14: a + 2 * k */
    leaq (%r14, %r8, 4), %r15    /* r15: a + 3 * k */
.SK.X64.FMA.M4N24.L2:
    SK_X64_FMA_M4N24_CP          /* inner loop */
    addq $384, %rbx              /* b += 24 * 4 */
    addq $4, %rax                /* index += 4 */
    cmpq %r8, %rax               /* index < k */
    jne .SK.X64.FMA.M4N24.L2
    leaq (%r15, %r8, 4), %rdi    /* rdi: a += 4 * k */
    SK_X64_AVX_M4N24_ST          /* store SIMD regs back to c */
    addq $384, %rdx
    subq $4, %rcx
    jne .SK.X64.FMA.M4N24.L1
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    retq


#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <string>
#include <iostream>

using std::string;
using std::cout;

#define BIT_TEST(bit_map, pos) (((bit_map) & (0x1 << (pos))) ? 1 : 0)
#define SET_FEAT(feat_mask) { feat |= (feat_mask); }

typedef enum
{
    _CPUID_X86_SSE_         = 0x1,
    _CPUID_X86_AVX_         = 0x2,
    _CPUID_X86_FMA_         = 0x4,
    _CPUID_X86_AVX512F_     = 0x8,
    _CPUID_X86_AVX512_VNNI_ = 0x10,
} cpuid_x86_feature_t;

struct cpuid_t
{
    unsigned int ieax;
    unsigned int iecx;
    unsigned int eax;
    unsigned int ebx;
    unsigned int ecx;
    unsigned int edx;
};

static unsigned int feat;

static void cpuid_x86_exec(struct cpuid_t *cpuid)
{
    asm volatile("pushq %%rbx\n"
        "cpuid\n"
        "movl %%ebx, %1\n"
        "popq %%rbx\n"
        : "=a"(cpuid->eax), "=r"(cpuid->ebx), "=c"(cpuid->ecx), "=d"(cpuid->edx)
        : "a"(cpuid->ieax), "c"(cpuid->iecx)
        : "cc");
}

static void cpuid_x86_init()
{
    struct cpuid_t cpuid;

    feat = 0;

    cpuid.ieax = 0x1;
    cpuid.iecx = 0x0;
    cpuid_x86_exec(&cpuid);

    if (BIT_TEST(cpuid.edx, 25))
    {
        SET_FEAT(_CPUID_X86_SSE_);
    }
    if (BIT_TEST(cpuid.ecx, 28))
    {
        SET_FEAT(_CPUID_X86_AVX_);
    }
    if (BIT_TEST(cpuid.ecx, 12))
    {
        SET_FEAT(_CPUID_X86_FMA_);
    }

    cpuid.ieax = 0x7;
    cpuid.iecx = 0x0;
    cpuid_x86_exec(&cpuid);

    if (BIT_TEST(cpuid.ebx, 16))
    {
        SET_FEAT(_CPUID_X86_AVX512F_);
    }
    if (BIT_TEST(cpuid.ecx, 11))
    {
        SET_FEAT(_CPUID_X86_AVX512_VNNI_);
    }
}

unsigned int cpuid_x86_support(cpuid_x86_feature_t feature)
{
    return feat & feature;
}

int main()
{
    cpuid_x86_init();

    string smtl_cmd = "gcc -pthread -O3 -c smtl.c\n";
    string asm_cmd = "";
    string c_cmd = "gcc -pthread ";
    string lnk_cmd = "gcc -pthread -lrt -o cpufp smtl.o cpufp_x86.o";

    string isa_macro = "";

    if (cpuid_x86_support(_CPUID_X86_AVX512F_))
    {
        isa_macro += "-D_AVX512F_ ";
        asm_cmd += "as -o cpufp_kernel_x86_avx512f.o cpufp_kernel_x86_avx512f.s\n";
        lnk_cmd += " cpufp_kernel_x86_avx512f.o";
    }
    if (cpuid_x86_support(_CPUID_X86_AVX512_VNNI_))
    {
        isa_macro += "-D_AVX512_VNNI_ ";
        asm_cmd += "as -o cpufp_kernel_x86_avx512_vnni.o cpufp_kernel_x86_avx512_vnni.s\n";
        lnk_cmd += " cpufp_kernel_x86_avx512_vnni.o";
    }
    if (cpuid_x86_support(_CPUID_X86_FMA_))
    {
        isa_macro += "-D_FMA_ ";
        asm_cmd += "as -o cpufp_kernel_x86_fma.o cpufp_kernel_x86_fma.s\n";
        lnk_cmd += " cpufp_kernel_x86_fma.o";
    }
    if (cpuid_x86_support(_CPUID_X86_AVX_))
    {
        isa_macro += "-D_AVX_ ";
        asm_cmd += "as -o cpufp_kernel_x86_avx.o cpufp_kernel_x86_avx.s\n";
        lnk_cmd += " cpufp_kernel_x86_avx.o";
    }
    if (cpuid_x86_support(_CPUID_X86_SSE_))
    {
        isa_macro += "-D_SSE_ ";
        asm_cmd += "as -o cpufp_kernel_x86_sse.o cpufp_kernel_x86_sse.s\n";
        lnk_cmd += " cpufp_kernel_x86_sse.o";
    }
    c_cmd += (isa_macro + "-c cpufp_x86.c\n");
    lnk_cmd += "\n";

    cout << smtl_cmd << asm_cmd << c_cmd << lnk_cmd;
    
    return 0;
}


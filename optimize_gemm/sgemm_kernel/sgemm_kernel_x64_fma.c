#include <immintrin.h>

void sgemm_kernel_x64_fma_m4n24(float *a,
    float *b,
    float *c,
    int m,
    int k)
{
    __m256 vc[12];
    __m256 vb[3], va;

    int i, j;
    int ii, jj;
    for (i = 0; i < m; i += 4)
    {
        float *pb = b;
        vc[0]  = _mm256_load_ps(c +  0);
        vc[1]  = _mm256_load_ps(c +  8);
        vc[2]  = _mm256_load_ps(c + 16);
        vc[3]  = _mm256_load_ps(c + 24);
        vc[4]  = _mm256_load_ps(c + 32);
        vc[5]  = _mm256_load_ps(c + 40);
        vc[6]  = _mm256_load_ps(c + 48);
        vc[7]  = _mm256_load_ps(c + 56);
        vc[8]  = _mm256_load_ps(c + 64);
        vc[9]  = _mm256_load_ps(c + 72);
        vc[10] = _mm256_load_ps(c + 80);
        vc[11] = _mm256_load_ps(c + 88);
        
        for (j = 0; j < k; j += 4)
        {
            vb[0] = _mm256_load_ps(pb +  0);
            vb[1] = _mm256_load_ps(pb +  8);
            vb[2] = _mm256_load_ps(pb + 16);
            va = _mm256_broadcast_ss(a + 0 * k + j + 0);
            vc[0]  = _mm256_fmadd_ps(va, vb[0], vc[0]);
            vc[1]  = _mm256_fmadd_ps(va, vb[1], vc[1]);
            vc[2]  = _mm256_fmadd_ps(va, vb[2], vc[2]);
            va = _mm256_broadcast_ss(a + 1 * k + j + 0);
            vc[3]  = _mm256_fmadd_ps(va, vb[0], vc[3]);
            vc[4]  = _mm256_fmadd_ps(va, vb[1], vc[4]);
            vc[5]  = _mm256_fmadd_ps(va, vb[2], vc[5]);
            va = _mm256_broadcast_ss(a + 2 * k + j + 0);
            vc[6]  = _mm256_fmadd_ps(va, vb[0], vc[6]);
            vc[7]  = _mm256_fmadd_ps(va, vb[1], vc[7]);
            vc[8]  = _mm256_fmadd_ps(va, vb[2], vc[8]);
            va = _mm256_broadcast_ss(a + 3 * k + j + 0);
            vc[9]  = _mm256_fmadd_ps(va, vb[0], vc[9]);
            vc[10] = _mm256_fmadd_ps(va, vb[1], vc[10]);
            vc[11] = _mm256_fmadd_ps(va, vb[2], vc[11]);
            
            vb[0] = _mm256_load_ps(pb + 24);
            vb[1] = _mm256_load_ps(pb + 32);
            vb[2] = _mm256_load_ps(pb + 40);
            va = _mm256_broadcast_ss(a + 0 * k + j + 1);
            vc[0]  = _mm256_fmadd_ps(va, vb[0], vc[0]);
            vc[1]  = _mm256_fmadd_ps(va, vb[1], vc[1]);
            vc[2]  = _mm256_fmadd_ps(va, vb[2], vc[2]);
            va = _mm256_broadcast_ss(a + 1 * k + j + 1);
            vc[3]  = _mm256_fmadd_ps(va, vb[0], vc[3]);
            vc[4]  = _mm256_fmadd_ps(va, vb[1], vc[4]);
            vc[5]  = _mm256_fmadd_ps(va, vb[2], vc[5]);
            va = _mm256_broadcast_ss(a + 2 * k + j + 1);
            vc[6]  = _mm256_fmadd_ps(va, vb[0], vc[6]);
            vc[7]  = _mm256_fmadd_ps(va, vb[1], vc[7]);
            vc[8]  = _mm256_fmadd_ps(va, vb[2], vc[8]);
            va = _mm256_broadcast_ss(a + 3 * k + j + 1);
            vc[9]  = _mm256_fmadd_ps(va, vb[0], vc[9]);
            vc[10] = _mm256_fmadd_ps(va, vb[1], vc[10]);
            vc[11] = _mm256_fmadd_ps(va, vb[2], vc[11]);
            
            vb[0] = _mm256_load_ps(pb + 48);
            vb[1] = _mm256_load_ps(pb + 56);
            vb[2] = _mm256_load_ps(pb + 64);
            va = _mm256_broadcast_ss(a + 0 * k + j + 2);
            vc[0]  = _mm256_fmadd_ps(va, vb[0], vc[0]);
            vc[1]  = _mm256_fmadd_ps(va, vb[1], vc[1]);
            vc[2]  = _mm256_fmadd_ps(va, vb[2], vc[2]);
            va = _mm256_broadcast_ss(a + 1 * k + j + 2);
            vc[3]  = _mm256_fmadd_ps(va, vb[0], vc[3]);
            vc[4]  = _mm256_fmadd_ps(va, vb[1], vc[4]);
            vc[5]  = _mm256_fmadd_ps(va, vb[2], vc[5]);
            va = _mm256_broadcast_ss(a + 2 * k + j + 2);
            vc[6]  = _mm256_fmadd_ps(va, vb[0], vc[6]);
            vc[7]  = _mm256_fmadd_ps(va, vb[1], vc[7]);
            vc[8]  = _mm256_fmadd_ps(va, vb[2], vc[8]);
            va = _mm256_broadcast_ss(a + 3 * k + j + 2);
            vc[9]  = _mm256_fmadd_ps(va, vb[0], vc[9]);
            vc[10] = _mm256_fmadd_ps(va, vb[1], vc[10]);
            vc[11] = _mm256_fmadd_ps(va, vb[2], vc[11]);
            
            vb[0] = _mm256_load_ps(pb + 72);
            vb[1] = _mm256_load_ps(pb + 80);
            vb[2] = _mm256_load_ps(pb + 88);
            va = _mm256_broadcast_ss(a + 0 * k + j + 3);
            vc[0]  = _mm256_fmadd_ps(va, vb[0], vc[0]);
            vc[1]  = _mm256_fmadd_ps(va, vb[1], vc[1]);
            vc[2]  = _mm256_fmadd_ps(va, vb[2], vc[2]);
            va = _mm256_broadcast_ss(a + 1 * k + j + 3);
            vc[3]  = _mm256_fmadd_ps(va, vb[0], vc[3]);
            vc[4]  = _mm256_fmadd_ps(va, vb[1], vc[4]);
            vc[5]  = _mm256_fmadd_ps(va, vb[2], vc[5]);
            va = _mm256_broadcast_ss(a + 2 * k + j + 3);
            vc[6]  = _mm256_fmadd_ps(va, vb[0], vc[6]);
            vc[7]  = _mm256_fmadd_ps(va, vb[1], vc[7]);
            vc[8]  = _mm256_fmadd_ps(va, vb[2], vc[8]);
            va = _mm256_broadcast_ss(a + 3 * k + j + 3);
            vc[9]  = _mm256_fmadd_ps(va, vb[0], vc[9]);
            vc[10] = _mm256_fmadd_ps(va, vb[1], vc[10]);
            vc[11] = _mm256_fmadd_ps(va, vb[2], vc[11]);
            
            pb += 96;
        }

        _mm256_store_ps(c +  0, vc[0]);
        _mm256_store_ps(c +  8, vc[1]);
        _mm256_store_ps(c + 16, vc[2]);
        _mm256_store_ps(c + 24, vc[3]);
        _mm256_store_ps(c + 32, vc[4]);
        _mm256_store_ps(c + 40, vc[5]);
        _mm256_store_ps(c + 48, vc[6]);
        _mm256_store_ps(c + 56, vc[7]);
        _mm256_store_ps(c + 64, vc[8]);
        _mm256_store_ps(c + 72, vc[9]);
        _mm256_store_ps(c + 80, vc[10]);
        _mm256_store_ps(c + 88, vc[11]);

        a += 4 * k;
        c += 96;
    }
}


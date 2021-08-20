#include <stdio.h>

/* Block sizes */
#define mc 256
#define kc 128

/* Create macros so that the matrices are stored in row-major order */

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

#define min(i, j) ((i) < (j) ? (i): (j))

/* Routine for computing C = A * B + C */

void PackMatrixB( int k, float *b, int ldb, float *b_to) 
{
  int j;
  for ( j = 0; j < k; ++j) {
    float *b_ij_pntr = &B(j, 0);
    *b_to++ = b_ij_pntr[0];
    *b_to++ = b_ij_pntr[1];
    *b_to++ = b_ij_pntr[2];
    *b_to++ = b_ij_pntr[3];
  }
}

void PackMatrixA( int k, float *a, int lda, float * a_to)
{
  int i;
  float
    *a_0i_pntr = a,
    *a_1i_pntr = a + lda,
    *a_2i_pntr = a + (lda << 1),
    *a_3i_pntr = a + (3 * lda);

  for (i = 0; i < k; ++i) {
    *a_to++ = *a_0i_pntr++;
    *a_to++ = *a_1i_pntr++;
    *a_to++ = *a_2i_pntr++;
    *a_to++ = *a_3i_pntr++;
  }
  
}

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

typedef union
{
  __m128 v;
  float d[4];
} v2df_t;

void AddDot4x4( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
  v2df_t c_p0_sum;
  v2df_t c_p1_sum;
  v2df_t c_p2_sum;
  v2df_t c_p3_sum;
  v2df_t  a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg, b_reg;

  c_p0_sum.v = _mm_setzero_ps();
  c_p1_sum.v = _mm_setzero_ps();
  c_p2_sum.v = _mm_setzero_ps();
  c_p3_sum.v = _mm_setzero_ps();
  a_0p_reg.v = _mm_setzero_ps();
  a_1p_reg.v = _mm_setzero_ps();
  a_2p_reg.v = _mm_setzero_ps();
  a_3p_reg.v = _mm_setzero_ps();

  for (int p = 0; p < k; ++p) {
    b_reg.v = _mm_load_ps(b);
    b += 4;

    a_0p_reg.v = _mm_set_ps1(a[0]);
    a_1p_reg.v = _mm_set_ps1(a[1]);
    a_2p_reg.v = _mm_set_ps1(a[2]);
    a_3p_reg.v = _mm_set_ps1(a[3]);
    a += 4;

    c_p0_sum.v += b_reg.v * a_0p_reg.v;
    c_p1_sum.v += b_reg.v * a_1p_reg.v;
    c_p2_sum.v += b_reg.v * a_2p_reg.v;
    c_p3_sum.v += b_reg.v * a_3p_reg.v;
  }

  C(0, 0) += c_p0_sum.d[0]; C(0, 1) += c_p0_sum.d[1];
  C(0, 2) += c_p0_sum.d[2]; C(0, 3) += c_p0_sum.d[3];

  C(1, 0) += c_p1_sum.d[0]; C(1, 1) += c_p1_sum.d[1];
  C(1, 2) += c_p1_sum.d[2]; C(1, 3) += c_p1_sum.d[3];

  C(2, 0) += c_p2_sum.d[0]; C(2, 1) += c_p2_sum.d[1];
  C(2, 2) += c_p2_sum.d[2]; C(2, 3) += c_p2_sum.d[3];

  C(3, 0) += c_p3_sum.d[0]; C(3, 1) += c_p3_sum.d[1];
  C(3, 2) += c_p3_sum.d[2]; C(3, 3) += c_p3_sum.d[3];
}

void InnerKernel( int m, int n, int k, float *a, int lda, 
                                       float *b, int ldb,
                                       float *c, int ldc )
{
  int i, j;
  float packedA[m * k];
  float packedB[k * n];

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
    PackMatrixB(k, &B(0, j), ldb, packedB + j * k);
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */
      if (0 == j) {
        PackMatrixA(k, &A(i, 0), lda, packedA + i * k);
      }
      AddDot4x4( k, packedA + i * k, k, packedB + j * k, 4, &C( i,j ), ldc );
    }
  }
}

void MY_MMult_4x4_13( int m, int n, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc ) 
{
  int i, p, pb, ib; 
  for (p = 0; p < k; p += kc) {
    pb = min(k - p, kc);
    for (i = 0; i < m; i += mc) {
      ib = min(m - i, mc);
      InnerKernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc);
    }
  }
}
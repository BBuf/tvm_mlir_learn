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
    *b_to++ = b_ij_pntr[4];
    *b_to++ = b_ij_pntr[5];
    *b_to++ = b_ij_pntr[6];
    *b_to++ = b_ij_pntr[7];
  }
}

void PackMatrixA( int k, float *a, int lda, float * a_to)
{
  int i;
  float
    *a_0i_pntr = a,
    *a_1i_pntr = a + lda,
    *a_2i_pntr = a + (lda << 1),
    *a_3i_pntr = a + (3 * lda),
    *a_4i_pntr = a + (4 * lda),
    *a_5i_pntr = a + (5 * lda),
    *a_6i_pntr = a + (6 * lda),
    *a_7i_pntr = a + (7 * lda);

  for (i = 0; i < k; ++i) {
    *a_to++ = *a_0i_pntr++;
    *a_to++ = *a_1i_pntr++;
    *a_to++ = *a_2i_pntr++;
    *a_to++ = *a_3i_pntr++;
    *a_to++ = *a_4i_pntr++;
    *a_to++ = *a_5i_pntr++;
    *a_to++ = *a_6i_pntr++;
    *a_to++ = *a_7i_pntr++;
  }
  
}

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
#include <immintrin.h> //avx2

typedef union
{
  __m256 v;
  float d[8];
} v2df_t;

void AddDot4x4( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
  v2df_t c_p0_sum;
  v2df_t c_p1_sum;
  v2df_t c_p2_sum;
  v2df_t c_p3_sum;
  v2df_t c_p4_sum;
  v2df_t c_p5_sum;
  v2df_t c_p6_sum;
  v2df_t c_p7_sum;
  v2df_t  a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg, b_reg;

  c_p0_sum.v = _mm256_setzero_ps();
  c_p1_sum.v = _mm256_setzero_ps();
  c_p2_sum.v = _mm256_setzero_ps();
  c_p3_sum.v = _mm256_setzero_ps();
  c_p4_sum.v = _mm256_setzero_ps();
  c_p5_sum.v = _mm256_setzero_ps();
  c_p6_sum.v = _mm256_setzero_ps();
  c_p7_sum.v = _mm256_setzero_ps();

  a_0p_reg.v = _mm256_setzero_ps();
  a_1p_reg.v = _mm256_setzero_ps();
  a_2p_reg.v = _mm256_setzero_ps();
  a_3p_reg.v = _mm256_setzero_ps();

  for (int p = 0; p < k; ++p) {
    // b_reg.v = _mm256_load_ps((float *)b);
    b_reg.v = _mm256_set_ps(b[7], b[6], b[5], b[4], b[3], b[2], b[1], b[0]);
    b += 8;

    a_0p_reg.v = _mm256_set_ps(a[0], a[0], a[0], a[0], a[0], a[0], a[0], a[0]);
    a_1p_reg.v = _mm256_set_ps(a[1], a[1], a[1], a[1], a[1], a[1], a[1], a[1]);
    a_2p_reg.v = _mm256_set_ps(a[2], a[2], a[2], a[2], a[2], a[2], a[2], a[2]);
    a_3p_reg.v = _mm256_set_ps(a[3], a[3], a[3], a[3], a[3], a[3], a[3], a[3]);
    
    // c_p0_sum.v += b_reg.v * a_0p_reg.v;
    // c_p1_sum.v += b_reg.v * a_1p_reg.v;
    // c_p2_sum.v += b_reg.v * a_2p_reg.v;
    // c_p3_sum.v += b_reg.v * a_3p_reg.v;

	  c_p0_sum.v = _mm256_fmadd_ps(b_reg.v, a_0p_reg.v, c_p0_sum.v);
    c_p1_sum.v = _mm256_fmadd_ps(b_reg.v, a_1p_reg.v, c_p1_sum.v);
    c_p2_sum.v = _mm256_fmadd_ps(b_reg.v, a_2p_reg.v, c_p2_sum.v);
    c_p3_sum.v = _mm256_fmadd_ps(b_reg.v, a_3p_reg.v, c_p3_sum.v);

    a_0p_reg.v = _mm256_set_ps(a[4], a[4], a[4], a[4], a[4], a[4], a[4], a[4]);
    a_1p_reg.v = _mm256_set_ps(a[5], a[5], a[5], a[5], a[5], a[5], a[5], a[5]);
    a_2p_reg.v = _mm256_set_ps(a[6], a[6], a[6], a[6], a[6], a[6], a[6], a[6]);
    a_3p_reg.v = _mm256_set_ps(a[7], a[7], a[7], a[7], a[7], a[7], a[7], a[7]);
    
    // c_p4_sum.v += b_reg.v * a_0p_reg.v;
    // c_p5_sum.v += b_reg.v * a_1p_reg.v;
    // c_p6_sum.v += b_reg.v * a_2p_reg.v;
    // c_p7_sum.v += b_reg.v * a_3p_reg.v;
	
	  c_p4_sum.v = _mm256_fmadd_ps(b_reg.v, a_0p_reg.v, c_p4_sum.v);
    c_p5_sum.v = _mm256_fmadd_ps(b_reg.v, a_1p_reg.v, c_p5_sum.v);
    c_p6_sum.v = _mm256_fmadd_ps(b_reg.v, a_2p_reg.v, c_p6_sum.v);
    c_p7_sum.v = _mm256_fmadd_ps(b_reg.v, a_3p_reg.v, c_p7_sum.v);
    a += 8;

  }

  C(0, 0) += c_p0_sum.d[0]; C(0, 1) += c_p0_sum.d[1]; 
  C(0, 2) += c_p0_sum.d[2]; C(0, 3) += c_p0_sum.d[3];
  C(0, 4) += c_p0_sum.d[4]; C(0, 5) += c_p0_sum.d[5]; 
  C(0, 6) += c_p0_sum.d[6]; C(0, 7) += c_p0_sum.d[7];
  

  C(1, 0) += c_p1_sum.d[0]; C(1, 1) += c_p1_sum.d[1];
  C(1, 2) += c_p1_sum.d[2]; C(1, 3) += c_p1_sum.d[3];
  C(1, 4) += c_p1_sum.d[4]; C(1, 5) += c_p1_sum.d[5];
  C(1, 6) += c_p1_sum.d[6]; C(1, 7) += c_p1_sum.d[7];

  C(2, 0) += c_p2_sum.d[0]; C(2, 1) += c_p2_sum.d[1];
  C(2, 2) += c_p2_sum.d[2]; C(2, 3) += c_p2_sum.d[3];
  C(2, 4) += c_p2_sum.d[4]; C(2, 5) += c_p2_sum.d[5];
  C(2, 6) += c_p2_sum.d[6]; C(2, 7) += c_p2_sum.d[7];

  C(3, 0) += c_p3_sum.d[0]; C(3, 1) += c_p3_sum.d[1];
  C(3, 2) += c_p3_sum.d[2]; C(3, 3) += c_p3_sum.d[3];
  C(3, 4) += c_p3_sum.d[4]; C(3, 5) += c_p3_sum.d[5];
  C(3, 6) += c_p3_sum.d[6]; C(3, 7) += c_p3_sum.d[7];

  C(4, 0) += c_p4_sum.d[0]; C(4, 1) += c_p4_sum.d[1]; 
  C(4, 2) += c_p4_sum.d[2]; C(4, 3) += c_p4_sum.d[3];
  C(4, 4) += c_p4_sum.d[4]; C(4, 5) += c_p4_sum.d[5]; 
  C(4, 6) += c_p4_sum.d[6]; C(4, 7) += c_p4_sum.d[7];
  

  C(5, 0) += c_p5_sum.d[0]; C(5, 1) += c_p5_sum.d[1];
  C(5, 2) += c_p5_sum.d[2]; C(5, 3) += c_p5_sum.d[3];
  C(5, 4) += c_p5_sum.d[4]; C(5, 5) += c_p5_sum.d[5];
  C(5, 6) += c_p5_sum.d[6]; C(5, 7) += c_p5_sum.d[7];

  C(6, 0) += c_p6_sum.d[0]; C(6, 1) += c_p6_sum.d[1];
  C(6, 2) += c_p6_sum.d[2]; C(6, 3) += c_p6_sum.d[3];
  C(6, 4) += c_p6_sum.d[4]; C(6, 5) += c_p6_sum.d[5];
  C(6, 6) += c_p6_sum.d[6]; C(6, 7) += c_p6_sum.d[7];

  C(7, 0) += c_p7_sum.d[0]; C(7, 1) += c_p7_sum.d[1];
  C(7, 2) += c_p7_sum.d[2]; C(7, 3) += c_p7_sum.d[3];
  C(7, 4) += c_p7_sum.d[4]; C(7, 5) += c_p7_sum.d[5];
  C(7, 6) += c_p7_sum.d[6]; C(7, 7) += c_p7_sum.d[7];
}

void InnerKernel( int m, int n, int k, float *a, int lda, 
                                       float *b, int ldb,
                                       float *c, int ldc )
{
  int i, j;
  float packedA[m * k];
  float packedB[k * n];

  for ( j=0; j<n; j+=8 ){        /* Loop over the columns of C, unrolled by 4 */
    PackMatrixB(k, &B(0, j), ldb, packedB + j * k);
    for ( i=0; i<m; i+=8 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */
      if (0 == j) {
        PackMatrixA(k, &A(i, 0), lda, packedA + i * k);
      }
      AddDot4x4( k, packedA + i * k, k, packedB + j * k, ldb, &C( i,j ), ldc );
    }
  }
}

// AddDot4x4( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )

void MY_MMult_4x4_14( int m, int n, int k, float *a, int lda, 
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
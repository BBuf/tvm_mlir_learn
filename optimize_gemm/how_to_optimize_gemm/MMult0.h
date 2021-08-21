#include <stdio.h>

#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#define C( i, j ) c[ (i)*ldc + (j) ]
// gemm C = A * B + C
void MatrixMultiply(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    for(int i = 0; i < m; i++){
        for (int p=0; p<k; p++ ){ 
            for (int j=0; j<n; j++ ){    
                C(i, j) = C(i, j) + A(i, p) * B(p, j);
            }
        }
    }
}
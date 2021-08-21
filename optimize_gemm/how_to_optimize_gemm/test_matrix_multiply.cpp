#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "MMult0.h"
// #include "MMult1.h"
// #include "MMult2.h"
// #include "MMult_1x4_3.h"
// #include "MMult_1x4_4.h"
// #include "MMult_1x4_5.h"
// #include "MMult_1x4_6.h"
// #include "MMult_1x4_7.h"
// #include "MMult_1x4_8.h"
// #include "MMult_1x4_9.h"
// #include "MMult_4x4_3.h"
// #include "MMult_4x4_4.h"
// #include "MMult_4x4_5.h"
// #include "MMult_4x4_6.h"
// #include "MMult_4x4_7.h"
#include "MMult_4x4_8.h"
// #include "MMult_4x4_10.h"
// #include "MMult_4x4_11.h"
// #include "MMult_4x4_12.h"
// #include "MMult_4x4_13.h"
// #include "MMult_4x4_14.h"

#include "dclock.h"
using namespace std;

#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#define abs( x ) ( (x) < 0.0 ? -(x) : (x) )

void random_matrix( int m, int n, float *a, int lda )
{
  double drand48();

  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
        A(i, j) = (float)drand48();
    }
  }
}

void copy_matrix(int m, int n, float *a, int lda, float *b, int ldb)
{
  int i, j;

  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
        B(i, j) = A(i, j);
    }
  }
}

float compare_matrices( int m, int n, float *a, int lda, float *b, int ldb )
{

  float max_diff = 0.0, diff;
  for (int i=0; i<m; i++ ){
    for (int j=0; j<n; j++ ){
        diff = abs(A(i, j) - B(i, j));
        
        max_diff = max(diff, max_diff);

        if(max_diff > 0.5f || max_diff < -0.5f) {
            printf("\n error: i %d  j %d diff %f", i, j, max_diff);
        }
    }
  }
  return max_diff;
}

static double get_time(struct timespec *start,
                       struct timespec *end) {
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

int m, n, k, lda, ldb, ldc;

double time_tmp, time_best, gflops, diff;

float *a, *b, *c, *prec, *nowc;    

int main(){

    struct timespec start, end;

    double time_used = 0.0;

    for(int i = 40; i <= 500; i += 40){
        m = i;
        n = i;
        k = i;
        gflops = 2.0 * m * n * k * 1.0e-09;
        lda = m;
        ldb = k;
        ldc = m;
        a = (float *)malloc(lda * k * sizeof(float));
        b = (float *)malloc(ldb * n * sizeof(float));
        c = (float *)malloc(ldc * n * sizeof(float));
        prec = (float *)malloc(ldc * n * sizeof(float));
        nowc = (float *)malloc(ldc * n * sizeof(float));
        // 随机填充矩阵
        random_matrix(m, k, a, lda);
        random_matrix(k, n, b, ldb);
        random_matrix(m, n, prec, ldc);

        memset(prec, 0, ldc * n * sizeof(float));

        copy_matrix(m, n, prec, ldc, nowc, ldc);

        // 以nowc为基准，判断矩阵运行算结果是否正确
        MatrixMultiply(m, n, k, a, lda, b, ldb, nowc, ldc);

        // 循环20次，以最快的运行时间为结果
        for(int j=0; j < 20; j++){
            
            copy_matrix(m, n, prec, ldc, c, ldc);

            clock_gettime(CLOCK_MONOTONIC_RAW, &start);

            MY_MMult_4x4_8(m, n, k, a, lda, b, ldb, c, ldc);

            clock_gettime(CLOCK_MONOTONIC_RAW, &end);

            time_tmp = get_time(&start, &end);
            
            if(j == 0)
                time_best = time_tmp;
            else
                time_best = min(time_best, time_tmp);
        }

        diff = compare_matrices(m, n, c, ldc, nowc, ldc);

        if(diff > 0.5f || diff < -0.5f){
            exit(0);
        }

        printf("%d %le %le \n", i, gflops / time_best, diff);
        fflush(stdout);

        free(a);
        free(b);
        free(c);
        free(prec);
        free(nowc);
    }
    printf("\n");
    fflush(stdout);
    return 0;
}

#include <stdio.h>

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Routine for computing C = A * B + C */

void AddDot1x4( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
  /* So, this routine computes four elements of C: 
           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 
           C( i, j ), C( i, j+1 ), C( i, j+2 ), C( i, j+3 ) 
	  
     in the original matrix C.
     In this version, we merge the four loops, computing four inner
     products simultaneously. */

  int p;

  //  AddDot( k, &A( 0, 0 ), lda, &B( 0, 0 ), &C( 0, 0 ) );
  //  AddDot( k, &A( 0, 0 ), lda, &B( 0, 1 ), &C( 0, 1 ) );
  //  AddDot( k, &A( 0, 0 ), lda, &B( 0, 2 ), &C( 0, 2 ) );
  //  AddDot( k, &A( 0, 0 ), lda, &B( 0, 3 ), &C( 0, 3 ) );
  for ( p=0; p<k; p++ ){
    C( 0, 0 ) += A( 0, p ) * B( p, 0 );     
    C( 0, 1 ) += A( 0, p ) * B( p, 1 );     
    C( 0, 2 ) += A( 0, p ) * B( p, 2 );     
    C( 0, 3 ) += A( 0, p ) * B( p, 3 );     
  }
}

void MY_MMult_1x4_5( int m, int n, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
    for ( i=0; i<m; i+=1 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */

      AddDot1x4( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
    }
  }
}



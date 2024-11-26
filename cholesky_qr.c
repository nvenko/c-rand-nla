/*
 * Copyright (c) 2024 Nicolas Venkovic
 * 
 * This file is part of c-rand-nla, currently a private repository.
 * 
 * This file is licensed under the MIT License.
 * For the full license text, see the LICENSE file in the root directory of this project.
 */

#include "rand_nla.h"

/*
void colRCholeskyQR(int n, int m, int p, int k, double *X, dSRHT *srht, double *Q, double *R, double *S)
Computes a QR decomposition by block column-based randomized Cholesky QR decompostion.

In:
  - int n: Dimension of the vectors.
  - int m: Number of vectors to orthogonalize.
  - int p: Block size.
  - int k: Sketching dimension.
  - double *X: m n-dimensional vectors assumed linearly independent and stored by column.
  - dSRHT *srht: SRHT data structure.

Out:
  - double *Q: m n-dimensional Theta-orthonormal vectors stored by column.
  - double *R: m-by-m R factor of the QR decomposition stored by column.
  - double *S: m k-dimensional orthonormal vectors stored by column.
*/
void colRCholeskyQR(int n, int m, int p, int k, double *X, dSRHT *srht, double *Q, double *R, double *S) {
  int s = m / p;
  double *P = (double*)mkl_malloc(k * s * sizeof(double), sizeof(double));
  double *Rtmp = (double*)mkl_malloc(m * s * sizeof(double), sizeof(double));
  double *StS = (double*)mkl_malloc(m * m * sizeof(double), sizeof(double));
  lapack_int *ipiv = (lapack_int*)mkl_malloc(m * sizeof(lapack_int), sizeof(lapack_int));
  BlockMatrixFreeTheta(&X[0], srht, s, P);
  MGS(k, s, P, &S[0], Rtmp);
  for (int v=0; v<s; v++)
    for (int u=0; u<s; u++)
      R[v * m + u] = Rtmp[v * s + u];
  for (int v=0; v<s; v++)
    for (int u=s; u<m; u++)
      R[v * m + u] = 0.;
  cblas_dcopy(n * s, &X[0], 1, &Q[0], 1);
  LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N', s, n, Rtmp, s, &Q[0], n);
  //BlockMatrixFreeTheta(&Q[0], srht, s, &S[0]);
  for (int i=1; i<p; i++) {
    BlockMatrixFreeTheta(&X[i * s * n], srht, s, P);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, i * s, i * s, k, 1., S, k, S, k, 0., StS, i * s);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, i * s, s, k, 1., S, k, P, k, 0., Rtmp, i * s);
    LAPACKE_dsysv(LAPACK_COL_MAJOR, 'U', i * s, s , StS, i * s, ipiv, Rtmp, i * s);
    for (int v=0; v<s; v++)
      for (int u=0; u<i*s; u++)
        R[i * s * m + v * m + u] = Rtmp[v * i * s + u];
    cblas_dcopy(n * s, &X[i * s * n], 1, &Q[i * s * n], 1);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, s, i * s, -1., Q, n, Rtmp, i * s, 1., &Q[i * s * n], n);
    MGS(k, s, &S[i * s * k], &S[i * s * k], Rtmp);
    for (int v=0; v<s; v++)
      for (int u=0; u<s; u++)
        R[i * s * m + v * m + i * s + u] = Rtmp[v * s + u];
    for (int v=0; v<s; v++)
      for (int u=(i+1)*s; u<m; u++)
        R[i * s * m + v * m + u] = 0.;
    LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N', s, n, Rtmp, s, &Q[i * s * n], n);
    //BlockMatrixFreeTheta(&Q[i * s * n], srht, s, &S[i * s * n]);
  }
  mkl_free(P);
  mkl_free(Rtmp);
  mkl_free(StS);
  mkl_free(ipiv);
}
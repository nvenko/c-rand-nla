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
void BMGS(int n, int m, double *X, double *Q, double *R)
Computes the block modified Gram-Schmidt orthogonalization of m n-vectors.

In:
  - int n: Dimension of the vectors.
  - int m: Number of vectors to orthogonalize.
  - double *X: m n-dimensional vectors assumed linearly independent and stored by column.

Out:
  - double *Q: m n-dimensional Theta-orthonormal vectors stored by column.
  - double *R: m-by-m R factor of the QR decomposition stored by column.
*/
void BMGS(int n, int m, int p, double *X, double *Q, double *R) {
  int s = m / p;
  double *Rtmp = (double*)mkl_malloc(s * s * sizeof(double), sizeof(double));
  MGS(n, s, &X[0], &Q[0], Rtmp);
  for (int v=0; v<s; v++)
    for (int u=0; u<s; u++)
      R[v * m + u] = Rtmp[v * s + u];
  for (int v=0; v<s; v++)
    for (int u=s; u<m; u++)
      R[v * m + u] = 0.;
  for (int i=1; i<p; i++) {
    cblas_dcopy(n * s, &X[i * s * n], 1, &Q[i * s * n], 1);
    for (int j=0; j<i; j++) {
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, s, s, n, 1., &Q[j * s * n], n, &Q[i * s * n], n, 0., Rtmp, s);
      for (int v=0; v<s; v++)
        for (int u=0; u<s; u++)
          R[i * s * m + v * m + j * s + u] = Rtmp[v * s + u];  
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, s, s, -1., &Q[j * s * n], n, Rtmp, s, 1., &Q[i * s * n], n);
    }
    MGS(n, s, &Q[i * s * n], &Q[i * s * n], Rtmp);
    for (int v=0; v<s; v++)
      for (int u=0; u<s; u++)
        R[i * s * m + v * m + i * s + u] = Rtmp[v * s + u];
    for (int v=0; v<s; v++)
      for (int u=(i+1)*s; u<m; u++)
        R[i * s * m + v * m + u] = 0.;
  }
}

/*
void BCGS2(int n, int m, double *X, double *Q, double *R)
Computes the block classical Gram-Schmidt orthogonalization of m n-vectors with re-orthogonalization.

In:
  - int n: Dimension of the vectors.
  - int m: Number of vectors to orthogonalize.
  - double *X: m n-dimensional vectors assumed linearly independent and stored by column.

Out:
  - double *Q: m n-dimensional Theta-orthonormal vectors stored by column.
  - double *R: m-by-m R factor of the QR decomposition stored by column.
*/
void BCGS2(int n, int m, int p, double *X, double *Q, double *R) {
  int s = m / p;
  double *Rtmp = (double*)mkl_malloc(m * s * sizeof(double), sizeof(double));
  MGS(n, s, &X[0], &Q[0], Rtmp);
  for (int v=0; v<s; v++)
    for (int u=0; u<s; u++)
      R[v * m + u] = Rtmp[v * s + u];
  for (int v=0; v<s; v++)
    for (int u=s; u<m; u++)
      R[v * m + u] = 0.;
  for (int i=1; i<p; i++) {
    cblas_dcopy(n * s, &X[i * s * n], 1, &Q[i * s * n], 1);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, i * s, s, n, 1., Q, n, &X[i * s * n], n, 0., Rtmp, i * s);
    for (int v=0; v<s; v++)
      for (int u=0; u<i*s; u++)
        R[i * s * m + v * m + u] = Rtmp[v * i * s + u];
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, s, i * s, -1., Q, n, Rtmp, i * s, 1., &Q[i * s * n], n);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, i * s, s, n, 1., Q, n, &Q[i * s * n], n, 0., Rtmp, i * s);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, s, i * s, -1., Q, n, Rtmp, i * s, 1., &Q[i * s * n], n);
    MGS(n, s, &Q[i * s * n], &Q[i * s * n], Rtmp);
    for (int v=0; v<s; v++)
      for (int u=0; u<s; u++)
        R[i * s * m + v * m + i * s + u] = Rtmp[v * s + u];
    for (int v=0; v<s; v++)
      for (int u=(i+1)*s; u<m; u++)
        R[i * s * m + v * m + u] = 0.;
  } 
}

/*
void BRGS(int n, int m, int k, double *X, dSRHT *srht, double *Q, double *R, double *S)
Computes the block randomized Gram-Schmidt orthogonalization of m n-vectors.
The sketching is done by a matrix-free procedure.

In:
  - int n: Dimension of the vectors.
  - int m: Number of vectors to orthogonalize.
  - int k: Sketching dimension.
  - double *X: m n-dimensional vectors assumed linearly independent and stored by column.
  - dSRHT *srht: SRHT data structure.

Out:
  - double *Q: m n-dimensional Theta-orthonormal vectors stored by column.
  - double *R: m-by-m R factor of the QR decomposition stored by column.
  - double *S: m k-dimensional orthonormal vectors stored by column.
*/
void BRGS(int n, int m, int p, int k, double *X, dSRHT *srht, double *Q, double *R, double *S, bool ffht) {
  int s = m / p;
  double *P = (double*)mkl_malloc(k * s * sizeof(double), sizeof(double));
  double *Rtmp = (double*)mkl_malloc(m * s * sizeof(double), sizeof(double));
  double *StS = (double*)mkl_malloc(m * m * sizeof(double), sizeof(double));
  lapack_int *ipiv = (lapack_int*)mkl_malloc(m * sizeof(lapack_int), sizeof(lapack_int));
  RGS(n, s, k, &X[0], srht, &Q[0], Rtmp, &S[0], true, ffht);
  for (int v=0; v<s; v++)
    for (int u=0; u<s; u++)
      R[v * m + u] = Rtmp[v * s + u];
  for (int v=0; v<s; v++)
    for (int u=s; u<m; u++)
      R[v * m + u] = 0.;
  for (int i=1; i<p; i++) {
    if (ffht) {
      BlockMatrixFreeTheta_ffht(&X[i * s * n], srht, s, P);
    }
    else {
      BlockMatrixFreeTheta(&X[i * s * n], srht, s, P);
    }
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, i * s, i * s, k, 1., S, k, S, k, 0., StS, i * s);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, i * s, s, k, 1., S, k, P, k, 0., Rtmp, i * s);
    LAPACKE_dsysv(LAPACK_COL_MAJOR, 'U', i * s, s , StS, i * s, ipiv, Rtmp, i * s);
    for (int v=0; v<s; v++)
      for (int u=0; u<i*s; u++)
        R[i * s * m + v * m + u] = Rtmp[v * i * s + u];
    cblas_dcopy(n * s, &X[i * s * n], 1, &Q[i * s * n], 1);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, s, i * s, -1., Q, n, Rtmp, i * s, 1., &Q[i * s * n], n);
    RGS(n, s, k, &Q[i * s * n], srht, &Q[i * s * n], Rtmp, &S[i * s * k], true, ffht);
    for (int v=0; v<s; v++)
      for (int u=0; u<s; u++)
        R[i * s * m + v * m + i * s + u] = Rtmp[v * s + u];
    for (int v=0; v<s; v++)
      for (int u=(i+1)*s; u<m; u++)
        R[i * s * m + v * m + u] = 0.;
  }
  mkl_free(P);
  mkl_free(Rtmp);
  mkl_free(StS);
  mkl_free(ipiv);
}

/*
void BRGS_new(int n, int m, int k, double *X, dSRHT *srht, double *Q, double *R, double *S, bool grouped)
Computes the block randomized Gram-Schmidt orthogonalization of m n-vectors.
The sketching is done by a matrix-free procedure.

In:
  - int n: Dimension of the vectors.
  - int m: Number of vectors to orthogonalize.
  - int k: Sketching dimension.
  - double *X: m n-dimensional vectors assumed linearly independent and stored by column.
  - dSRHT *srht: SRHT data structure.

Out:
  - double *Q: m n-dimensional Theta-orthonormal vectors stored by column.
  - double *R: m-by-m R factor of the QR decomposition stored by column.
  - double *S: m k-dimensional orthonormal vectors stored by column.
*/
/*void BRGS_new(int n, int m, int p, int k, double *X, dBSRHT *Bsrht, dSRHT *srht, double *Q, double *R, double *S, bool grouped) {
  int s = m / p;
  double *P = (double*)mkl_malloc(k * s * sizeof(double), sizeof(double));
  double *Rtmp = (double*)mkl_malloc(m * s * sizeof(double), sizeof(double));
  double *StS = (double*)mkl_malloc(m * m * sizeof(double), sizeof(double));
  lapack_int *ipiv = (lapack_int*)mkl_malloc(m * sizeof(lapack_int), sizeof(lapack_int));
  RGS(n, s, k, &X[0], srht, &Q[0], Rtmp, &S[0], true, false);
  for (int v=0; v<s; v++)
    for (int u=0; u<s; u++)
      R[v * m + u] = Rtmp[v * s + u];
  for (int v=0; v<s; v++)
    for (int u=s; u<m; u++)
      R[v * m + u] = 0.;
  for (int i=1; i<p; i++) {
    if (grouped) {
      BlockMatrixFreeTheta_grouped(&X[i * s * n], Bsrht, P);
    }
    else {
      BlockMatrixFreeTheta(&X[i * s * n], srht, s, P);
    }
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, i * s, i * s, k, 1., S, k, S, k, 0., StS, i * s);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, i * s, s, k, 1., S, k, P, k, 0., Rtmp, i * s);
    LAPACKE_dsysv(LAPACK_COL_MAJOR, 'U', i * s, s , StS, i * s, ipiv, Rtmp, i * s);
    for (int v=0; v<s; v++)
      for (int u=0; u<i*s; u++)
        R[i * s * m + v * m + u] = Rtmp[v * i * s + u];
    cblas_dcopy(n * s, &X[i * s * n], 1, &Q[i * s * n], 1);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, s, i * s, -1., Q, n, Rtmp, i * s, 1., &Q[i * s * n], n);
    RGS(n, s, k, &Q[i * s * n], srht, &Q[i * s * n], Rtmp, &S[i * s * k], true, false);
    for (int v=0; v<s; v++)
      for (int u=0; u<s; u++)
        R[i * s * m + v * m + i * s + u] = Rtmp[v * s + u];
    for (int v=0; v<s; v++)
      for (int u=(i+1)*s; u<m; u++)
        R[i * s * m + v * m + u] = 0.;
  }
  mkl_free(P);
  mkl_free(Rtmp);
  mkl_free(StS);
  mkl_free(ipiv);
}*/

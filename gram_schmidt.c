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
void CGS(int n, int m, double *X, double *Q, double *R)
Computes the classical Gram-Schmidt orthogonalization of m n-vectors.

In:
  - int n: Dimension of the vectors.
  - int m: Number of vectors to orthogonalize.
  - double *X: m n-dimensional vectors assumed linearly independent
               and stored by columns in an unrolled column major array.

Out:
  - double *Q: Matrix Q of QR decomposition of X stored by columns in an 
               unrolled column major array.
  - double *R: Matrix R of QR decomposition of X stored by columns in an 
               unrolled column major array.
*/
void CGS(int n, int m, double *X, double *Q, double *R) {
  R[0] = sqrt(cblas_ddot(n, &X[0], 1, &X[0], 1));
  cblas_dcopy(n, &X[0], 1, &Q[0], 1);
  cblas_dscal(n, 1./R[0], &Q[0], 1);
  for (int j=1; j<m; j++)
    R[j] = 0.;
  for (int j=1; j<m; j++) {
    cblas_dgemv(CblasColMajor, CblasTrans, n, j, 1., &Q[0], n, &X[j * n], 1, 0., &R[j * m], 1);
    cblas_dcopy(n, &X[j * n], 1, &Q[j * n], 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, n, j, -1., &Q[0], n, &R[j * m], 1, 1., &Q[j * n], 1);
    R[j * m + j] = sqrt(cblas_ddot(n, &Q[j * n], 1, &Q[j * n], 1));
    cblas_dscal(n, 1./R[j * m + j], &Q[j * n], 1);
    for (int i=j+1; i<m; i++)
      R[j * m + i] = 0.;
  }
}

/*
void MGS(int n, int m, double *X, double *Q, double *R)
Computes the modified Gram-Schmidt orthogonalization of m n-vectors.

In:
  - int n: Dimension of the vectors.
  - int m: Number of vectors to orthogonalize.
  - double *X: m n-dimensional vectors assumed linearly independent
               and stored by column in an unrolled column major array.

Out:
  - double *Q: Matrix Q of QR decomposition of X stored by columns in an 
               unrolled column major array.
  - double *R: Matrix R of QR decomposition of X stored by columns in an 
               unrolled column major array.
*/
void MGS(int n, int m, double *X, double *Q, double *R) {
  R[0] = sqrt(cblas_ddot(n, &X[0], 1, &X[0], 1));
  cblas_dcopy(n, &X[0], 1, &Q[0], 1);
  cblas_dscal(n, 1./R[0], &Q[0], 1);
  for (int j=1; j<m; j++)
    R[j] = 0.;
  for (int j=1; j<m; j++) {
    cblas_dcopy(n, &X[j * n], 1, &Q[j * n], 1);
    for (int i=0; i<j; i++) {
      R[j * m + i] = cblas_ddot(n, &Q[j * n], 1, &Q[i * n], 1);
      cblas_daxpy(n, -R[j * m + i], &Q[i * n], 1, &Q[j * n], 1);
    }
    R[j * m + j] = sqrt(cblas_ddot(n, &Q[j * n], 1, &Q[j * n], 1));
    for (int i=j+1; i<m; i++)
      R[j * m + i] = 0.;
    cblas_dscal(n, 1./R[j * m + j], &Q[j * n], 1);
  }
}

/*
void CGS2(int n, int m, double *X, double *Q, double *R)
Computes the classical Gram-Schmidt orthogonalization with 
re-orthogonalization of m n-vectors.

In:
  - int n: Dimension of the vectors.
  - int m: Number of vectors to orthogonalize.
  - double *X: m n-dimensional vectors assumed linearly independent
               and stored by columns in an unrolled column major array.

Out:
  - double *Q: Matrix Q of QR decomposition of X stored by columns in an 
               unrolled column major array.
  - double *R: Matrix R of QR decomposition of X stored by columns in an 
               unrolled column major array.
*/
void CGS2(int n, int m, double *X, double *Q, double *R) {
  double *Rtmp = (double*)mkl_malloc(m * sizeof(double), sizeof(double));
  R[0] = sqrt(cblas_ddot(n, &X[0], 1, &X[0], 1));
  cblas_dcopy(n, &X[0], 1, &Q[0], 1);
  cblas_dscal(n, 1./R[0], &Q[0], 1);
  for (int j=1; j<m; j++)
    R[j] = 0.;
  for (int j=1; j<m; j++) {
    cblas_dgemv(CblasColMajor, CblasTrans, n, j, 1., &Q[0], n, &X[j * n], 1, 0., &R[j * m], 1);
    cblas_dcopy(n, &X[j * n], 1, &Q[j * n], 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, n, j, -1., &Q[0], n, &R[j * m], 1, 1., &Q[j * n], 1);
    cblas_dgemv(CblasColMajor, CblasTrans, n, j, 1., &Q[0], n, &Q[j * n], 1, 0., Rtmp, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, n, j, -1., &Q[0], n, Rtmp, 1, 1., &Q[j * n], 1);
    R[j * m + j] = sqrt(cblas_ddot(n, &Q[j * n], 1, &Q[j * n], 1));
    cblas_dscal(n, 1./R[j * m + j], &Q[j * n], 1);
    for (int i=j+1; i<m; i++)
      R[j * m + i] = 0.;
  }
}

/*
void RGS(int n, int m, int k, double *X, dSRHT *srht, double *Q, double *R, double *S)
Computes the randomized Gram-Schmidt orthogonalization of m n-vectors.
The sketching is done by a matrix-free procedure.

In:
  - int n: Dimension of the vectors.
  - int m: Number of vectors to orthogonalize.
  - double *X: m n-dimensional vectors assumed linearly independent
               and stored by column.
  - dSRHT *srht: SRHT data structure.

Out:
  - double *Q: Matrix Q of QR decomposition of X stored by columns in an 
               unrolled column major array.
  - double *R: Matrix R of QR decomposition of X stored by columns in an 
               unrolled column major array.
*/
void RGS(int n, int m, int k, double *X, dSRHT *srht, double *Q, double *R, double *S, bool blas2, bool ffht) {
  double *p = (double*)mkl_malloc(k * sizeof(double), sizeof(double));
  double *StS = (double*)mkl_malloc(m * m * sizeof(double), sizeof(double));
  //lapack_int *ipiv = (lapack_int*)mkl_malloc(m * sizeof(lapack_int), sizeof(lapack_int)); // uncomment for normal equations
  cblas_dcopy(n, &X[0], 1, &Q[0], 1);
  if (ffht) {
    dMatrixFreeTheta_ffht(&Q[0], srht, &S[0]);
  }
  else {
    dMatrixFreeTheta(&Q[0], srht, &S[0]);
  }
  R[0] = sqrt(cblas_ddot(k, &S[0], 1, &S[0], 1));
  cblas_dscal(k, 1./R[0], &S[0], 1);
  cblas_dscal(n, 1./R[0], &Q[0], 1);
  for (int j=1; j<m; j++)
    R[j] = 0.;
  for (int i=1; i<m; i++) {
    if (ffht) {
      dMatrixFreeTheta_ffht(&X[i * n], srht, p);
    }
    else {
      dMatrixFreeTheta(&X[i * n], srht, p);
    }
    Richardson_lsq(k, i, S, p, &R[i * m]); // comment for normal equations
    //cblas_dgemv(CblasColMajor, CblasTrans, k, i, 1., S, k, p, 1, 0., &R[i * m], 1); // uncomment for normal equations
    //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, i, i, k, 1., S, k, S, k, 0., StS, i); // uncomment for normal equations
    //LAPACKE_dsysv(LAPACK_COL_MAJOR, 'U', i, 1 , StS, i, ipiv, &R[i * m], i); // uncomment for normal equations
    cblas_dcopy(n, &X[i * n], 1, &Q[i * n], 1);
    if (blas2) {
      cblas_dgemv(CblasColMajor, CblasNoTrans, n, i, -1., Q, n, &R[i * m], 1, 1., &Q[i * n], 1);
    }
    else {
      for (int j=0; j<i; j++) {
        cblas_daxpy(n, -R[i * m + j], &Q[j * n], 1, &Q[i * n], 1);
      }
    }
    if (ffht) {
      dMatrixFreeTheta_ffht(&Q[i * n], srht, &S[i * k]);
    }
    else {
      dMatrixFreeTheta(&Q[i * n], srht, &S[i * k]);
    }
    R[i * m + i] = sqrt(cblas_ddot(k, &S[i * k], 1, &S[i * k], 1));
    cblas_dscal(k, 1./R[i * m + i], &S[i * k], 1);
    cblas_dscal(n, 1./R[i * m + i], &Q[i * n], 1);
    for (int j=i+1; j<m; j++)
      R[i * m + j] = 0.;
  }
  mkl_free(p);
  mkl_free(StS);
  //mkl_free(ipiv); // uncomment for normal equations
}

/*
void RGS_mp(int n, int m, int k, double *X, dSRHT *srht, float *Q, double *R, double *S)
Computes the randomized Gram-Schmidt orthogonalization of m n-vectors.
The sketching is done by a matrix-free procedure.

In:
  - int n: Dimension of the vectors.
  - int m: Number of vectors to orthogonalize.
  - double *X: m n-dimensional vectors assumed linearly independent
               and stored by column.
  - dSRHT *srht: SRHT data structure.

Out:
  - float *Q: Matrix Q of QR decomposition of X stored by columns in an 
               unrolled column major array.
  - double *R: Matrix R of QR decomposition of X stored by columns in an 
               unrolled column major array.
*/
void RGS_mp(int n, int m, int k, double *X, dSRHT *srht, float *Q, double *R, double *S, bool ffht) {
  double *q = (double*)mkl_malloc(n * sizeof(double), sizeof(double));
  double *p = (double*)mkl_malloc(k * sizeof(double), sizeof(double));
  double *StS = (double*)mkl_malloc(m * m * sizeof(double), sizeof(double));
  float *r = (float*)mkl_malloc(m * sizeof(float), sizeof(float));
  //lapack_int *ipiv = (lapack_int*)mkl_malloc(m * sizeof(lapack_int), sizeof(lapack_int)); // uncomment for normal equations
  cblas_dcopy(n, &X[0], 1, q, 1);
  if (ffht) {
    dMatrixFreeTheta_ffht(q, srht, &S[0]);
  }
  else {
    dMatrixFreeTheta(q, srht, &S[0]);
  }
  R[0] = sqrt(cblas_ddot(k, &S[0], 1, &S[0], 1));
  cblas_dscal(k, 1./R[0], &S[0], 1);
  cblas_dscal(n, 1./R[0], q, 1);
  cblas_dscopy(n, q, &Q[0]);
  for (int j=1; j<m; j++)
    R[j] = 0.;
  for (int i=1; i<m; i++) {
    if (ffht) {
      dMatrixFreeTheta_ffht(&X[i * n], srht, p);
    }
    else {
      dMatrixFreeTheta(&X[i * n], srht, p);
    }
    Richardson_lsq(k, i, S, p, &R[i * m]); // comment for normal equations
    //cblas_dgemv(CblasColMajor, CblasTrans, k, i, 1., S, k, p, 1, 0., &R[i * m], 1); // uncomment for normal equations
    //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, i, i, k, 1., S, k, S, k, 0., StS, i); // uncomment for normal equations
    //LAPACKE_dsysv(LAPACK_COL_MAJOR, 'U', i, 1 , StS, i, ipiv, &R[i * m], i); // uncomment for normal equations
    cblas_dscopy(n, &X[i * n], &Q[i * n]);
    cblas_dscopy(i, &R[i * m], r);
    cblas_sgemv(CblasColMajor, CblasNoTrans, n, i, -1., Q, n, r, 1, 1., &Q[i * n], 1);
    cblas_sdcopy(n, &Q[i * n], q);
    if (ffht) {
      dMatrixFreeTheta_ffht(q, srht, &S[i * k]);
    }
    else {
      dMatrixFreeTheta(q, srht, &S[i * k]);

    }
    R[i * m + i] = sqrt(cblas_ddot(k, &S[i * k], 1, &S[i * k], 1));
    cblas_dscal(k, 1./R[i * m + i], &S[i * k], 1);
    cblas_dscal(n, 1./R[i * m + i], q, 1);
    cblas_dscopy(n, q, &Q[i * n]);
    for (int j=i+1; j<m; j++)
      R[i * m + j] = 0.;
  }
  mkl_free(p);
  mkl_free(StS);
  mkl_free(q);
  mkl_free(r);
  //mkl_free(ipiv); // uncomment for normal equations
}

/*
void RGS_cb(int n, int m, int k, double *X, dSRHT *srht, float *Q, double *R, double *S)
Computes the randomized Gram-Schmidt orthogonalization of m n-vectors.
The sketching is done by a matrix-free procedure.

In:
  - int n: Dimension of the vectors.
  - int m: Number of vectors to orthogonalize.
  - double *X: m n-dimensional vectors assumed linearly independent
               and stored by column.
  - dSRHT *srht: SRHT data structure.

Out:
  - float *Q: Matrix Q of QR decomposition of X stored by columns in an 
               unrolled column major array.
  - double *R: Matrix R of QR decomposition of X stored by columns in an 
               unrolled column major array.
*/
void RGS_cb(int n, int m, int k, double *X, dSRHT *srht, float *Q, double *R, double *S, bool ffht) {
  double *q = (double*)mkl_malloc(n * sizeof(double), sizeof(double));
  double *p = (double*)mkl_malloc(k * sizeof(double), sizeof(double));
  double *StS = (double*)mkl_malloc(m * m * sizeof(double), sizeof(double));
  //float *r = (float*)mkl_malloc(m * sizeof(float), sizeof(float));
  //lapack_int *ipiv = (lapack_int*)mkl_malloc(m * sizeof(lapack_int), sizeof(lapack_int)); // uncomment for normal equations
  //cblas_dcopy(n, &X[0], 1, &Q[0], 1);
  cblas_dcopy(n, &X[0], 1, q, 1);
  if (ffht) {
    dMatrixFreeTheta_ffht(q, srht, &S[0]);
  }
  else {
    dMatrixFreeTheta(q, srht, &S[0]);
  }
  R[0] = sqrt(cblas_ddot(k, &S[0], 1, &S[0], 1));
  cblas_dscal(k, 1./R[0], &S[0], 1);
  cblas_dscal(n, 1./R[0], q, 1);
  cblas_dscopy(n, q, &Q[0]);
  for (int j=1; j<m; j++)
    R[j] = 0.;
  for (int i=1; i<m; i++) {
    if (ffht) {
      dMatrixFreeTheta_ffht(&X[i * n], srht, p);
    }
    else {
      dMatrixFreeTheta(&X[i * n], srht, p);
    }
    Richardson_lsq(k, i, S, p, &R[i * m]); // comment for normal equations
    //cblas_dgemv(CblasColMajor, CblasTrans, k, i, 1., S, k, p, 1, 0., &R[i * m], 1); // uncomment for normal equations
    //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, i, i, k, 1., S, k, S, k, 0., StS, i); // uncomment for normal equations
    //LAPACKE_dsysv(LAPACK_COL_MAJOR, 'U', i, 1 , StS, i, ipiv, &R[i * m], i); // uncomment for normal equations
    //cblas_dscopy(n, &X[i * n], &Q[i * n]);
    //cblas_dscopy(i, &R[i * m], r);
    cblas_dcopy(n, &X[i * n], 1, q, 1);
    //cblas_sgemv(CblasColMajor, CblasNoTrans, n, i, -1., Q, n, r, 1, 1., &Q[i * n], 1);
    cblas_sdgemv(n, i, q, Q, &R[i * m]);

    //cblas_sdcopy(n, &Q[i * n], q);
    if (ffht) {
      dMatrixFreeTheta_ffht(q, srht, &S[i * k]);
    }
    else {
      dMatrixFreeTheta(q, srht, &S[i * k]);

    }
    R[i * m + i] = sqrt(cblas_ddot(k, &S[i * k], 1, &S[i * k], 1));
    cblas_dscal(k, 1./R[i * m + i], &S[i * k], 1);
    cblas_dscal(n, 1./R[i * m + i], q, 1);
    cblas_dscopy(n, q, &Q[i * n]);
    for (int j=i+1; j<m; j++)
      R[i * m + j] = 0.;
  }
  mkl_free(p);
  mkl_free(StS);
  mkl_free(q);
  //mkl_free(r);
  //mkl_free(ipiv); // uncomment for normal equations
}

/*
void RGS_low_sync(int n, int m, int k, double *X, dSRHT *srht, double *Q, double *R, double *S)
Computes the randomized Gram-Schmidt orthogonalization of m n-vectors while limiting the potential 
number of synchronizations to 1. The sketching is done by a matrix-free procedure.

In:
  - int n: Dimension of the vectors.
  - int m: Number of vectors to orthogonalize.
  - double *X: m n-dimensional vectors assumed linearly independent
               and stored by column.
  - dSRHT *srht: SRHT data structure.

Out:
  - double *Q: Matrix Q of QR decomposition of X stored by columns in an 
               unrolled column major array.
  - double *R: Matrix R of QR decomposition of X stored by columns in an 
               unrolled column major array.
*/
void RGS_low_sync(int n, int m, int k, double *X, dSRHT *srht, double *Q, double *R, double *S, bool ffht) {
  double *p = (double*)mkl_malloc(k * sizeof(double), sizeof(double));
  double *StS = (double*)mkl_malloc(m * m * sizeof(double), sizeof(double));
  //lapack_int *ipiv = (lapack_int*)mkl_malloc(m * sizeof(lapack_int), sizeof(lapack_int)); // uncomment for normal equations
  cblas_dcopy(n, &X[0], 1, &Q[0], 1);
  if (ffht) {
    dMatrixFreeTheta_ffht(&Q[0], srht, &S[0]);
  }
  else {
    dMatrixFreeTheta(&Q[0], srht, &S[0]);
  }
  R[0] = sqrt(cblas_ddot(k, &S[0], 1, &S[0], 1));
  cblas_dscal(k, 1./R[0], &S[0], 1);
  cblas_dscal(n, 1./R[0], &Q[0], 1);
  for (int j=1; j<m; j++)
    R[j] = 0.;
  for (int i=1; i<m; i++) {
    if (ffht) {
      dMatrixFreeTheta_ffht(&X[i * n], srht, p);
      dMatrixFreeTheta_ffht(&Q[(i - 1) * n], srht, &S[(i - 1) * k]);
    }
    else {
      dMatrixFreeTheta(&X[i * n], srht, p);
      dMatrixFreeTheta(&Q[(i - 1) * n], srht, &S[(i - 1) * k]);
    }
    Richardson_lsq(k, i, S, p, &R[i * m]); // comment for normal equations
    //cblas_dgemv(CblasColMajor, CblasTrans, k, i, 1., S, k, p, 1, 0., &R[i * m], 1); // uncomment for normal equations
    //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, i, i, k, 1., S, k, S, k, 0., StS, i); // uncomment for normal equations
    //LAPACKE_dsysv(LAPACK_COL_MAJOR, 'U', i, 1 , StS, i, ipiv, &R[i * m], i); // uncomment for normal equations
    cblas_dcopy(n, &X[i * n], 1, &Q[i * n], 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, n, i, -1., Q, n, &R[i * m], 1, 1., &Q[i * n], 1);
    cblas_dcopy(k, p, 1, &S[i * k], 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, k, i, -1., S, k, &R[i * m], 1, 1., &S[i * k], 1);
    R[i * m + i] = sqrt(cblas_ddot(k, &S[i * k], 1, &S[i * k], 1));
    cblas_dscal(k, 1./R[i * m + i], &S[i * k], 1);
    cblas_dscal(n, 1./R[i * m + i], &Q[i * n], 1);
    for (int j=i+1; j<m; j++)
      R[i * m + j] = 0.;
  }
  mkl_free(p);
  mkl_free(StS);
  //mkl_free(ipiv); // uncomment for normal equations
}
/*
 * Copyright (c) 2024 Nicolas Venkovic
 * 
 * This file is part of c-rand-nla, currently a private repository.
 * 
 * This file is licensed under the MIT License.
 * For the full license text, see the LICENSE file in the root directory of this project.
 */

#include "rand_nla.h"

void cblas_sdgemv(int m, int n, double *a, float *B, double *c) {
  // a = a - B * c
  for (int i=0; i<n; i++) {
    float *Bi = B + i * m;
    double ci = c[i];
    for (int j=0; j<m; j++) {
      double Bji_db = (double) Bi[j];
      a[j] -= Bji_db * ci;
    }
  }
}

/*
void Richardson_lsq(double *S, double *p, double *y)
Computes the least-squares solution by Richardson iteration
for an approximately orthonormal LHS.

In:
  - int k: Number of rows in S.
  - int i: Number of columns in S.
  - double *S: LHS of LSQ problem.
  - double *p: RHS of LSQ problem.

Out:
  - double *y: Solution of LSQ problem.
*/
void Richardson_lsq(int k, int i, double *S, double *p, double *y) {
  double *p2 = (double*)mkl_malloc(k * sizeof(double), sizeof(double));
  double *dy = (double*)mkl_malloc(i * sizeof(double), sizeof(double));
  for (int j=0; j<i; j++) y[j] = 0.;
  cblas_dcopy(k, p, 1, p2, 1);
  for (int j=0; j<5; j++) {
    cblas_dgemv(CblasColMajor, CblasTrans, k, i, 1., S, k, p2, 1, 0., dy, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, k, i, -1., S, k, dy, 1, 1., p2, 1);
    cblas_daxpy(i, 1., dy, 1, y, 1);
  }
  mkl_free(p2);
  mkl_free(dy);
}

/*
double matrix_2norm(double *A, int n, int m)
Computes an approximation of the 2-norm of the matrix A by the power method.

In:
  - double *A: n-by-m matrix stored in a vector by column.
  - int n: Number of rows of the matrix A.
  - int m: Number of columns of the matrix A.

Out:
  - double sigma: Approximate 2-norm of the matrix A.
*/
double matrix_2norm(double *A, int n, int m) {
  int maxit = 100;
  double *x = (double*)mkl_malloc(m * sizeof(double), sizeof(double));
  double *Ax = (double*)mkl_malloc(n * sizeof(double), sizeof(double));
  double *AtAx = (double*)mkl_malloc(m * sizeof(double), sizeof(double));
  double lbda, AtAx_nrm, x_nrm;
  for (int i=0; i<m; i++)
    x[i] = rand() / (double) RAND_MAX;
  for (int i=0; i<maxit; i++) {
    cblas_dgemv(CblasColMajor, CblasNoTrans, n, m, 1., A, n, x, 1, 0., Ax, 1);
    cblas_dgemv(CblasColMajor, CblasTrans, n, m, 1., A, n, Ax, 1, 0., AtAx, 1);
    AtAx_nrm = cblas_dnrm2(m, AtAx, 1);
    x_nrm = cblas_dnrm2(m, x, 1);
    lbda = AtAx_nrm / x_nrm;
    cblas_dcopy(m, AtAx, 1, x, 1);
    cblas_dscal(m, 1./AtAx_nrm, x, 1);    
  }
  mkl_free(x);
  mkl_free(Ax);
  mkl_free(AtAx);
  return sqrt(lbda);
}

/*
void cblas_dscopy(int n, double *X, float *Y)
Copies double *X to float *Y.

In:
  - int n: Vector dimension.
  - double *X: n-dimensional vector.

Out:
  - float *Y: n-dimensional vector
*/
void cblas_dscopy(int n, double *X, float *Y) {
  #pragma omp parallel for
  for (int i=0; i<n; i++) Y[i] = (float) X[i];
}

/*
void cblas_sdcopy(int n, float *X, double *Y)
Copies double *X to float *Y.

In:
  - int n: Vector dimension.
  - float *X: n-dimensional vector.

Out:
  - double *Y: n-dimensional vector
*/
void cblas_sdcopy(int n, float *X, double *Y) {
  #pragma omp parallel for
  for (int i=0; i<n; i++) Y[i] = (double) X[i];
}
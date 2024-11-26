/*
 * Copyright (c) 2024 Nicolas Venkovic
 * 
 * This file is part of rand-nla, currently a private repository.
 * 
 * This file is licensed under the MIT License.
 * For the full license text, see the LICENSE file in the root directory of this project.
 */

#ifndef RAND_NLA_H
#define RAND_NLA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include <stdbool.h>
#include <mkl.h>

/*
Subsampled double precision randomized Hadamard transform                
*/
typedef struct dSRHTs {
  int n;
  int N;
  int log2_N;
  int k;
  int *D;
  int *perm;
  double *z;
} dSRHT;

/*
Block subsampled double precision randomized Hadamard transform                
*/
/*typedef struct dBSRHTs {
  int n;
  int N;
  int log2_N;
  int k;
  int s;
  int *D;
  int *perm;
  double *Z;
} dBSRHT;*/

/*
Subsampled single precision randomized Hadamard transform                
*/
typedef struct sSRHTs {
  int n;
  int N;
  int log2_N;
  int k;
  int *D;
  int *perm;
  float *z;
} sSRHT;

// In sketching.c
dSRHT SetdSrht(int n, int k);
/*dBSRHT SetdBSrht(int n, int k, int s);*/
sSRHT SetsSrht(int n, int k);
int *RandomPermutation(int n);
void dfwht(double *x, int n, int *D, double *z, int N, int log2_N);
void sfwht(float *x, int n, int *D, float *z, int N, int log2_N);
void dMatrixFreeTheta(double *x, dSRHT *srht, double *Theta_x);
void dMatrixFreeTheta_ffht(double *x, dSRHT *srht, double *Theta_x);
void sMatrixFreeTheta(float *x, sSRHT *srht, float *Theta_x);
void sMatrixFreeTheta_ffht(float *x, sSRHT *srht, float *Theta_x);
void BlockMatrixFreeTheta(double *X, dSRHT *srht, int s, double *Theta_X);
/*void BlockMatrixFreeTheta_grouped(double *X, dBSRHT *srht, double *Theta_X);*/
void BlockMatrixFreeTheta_ffht(double *X, dSRHT *srht, int s, double *Theta_X);
void dumb_fht(double *buf, int log_n);

// In fht_avx.c
int fht_float(float *buf, int log_n);
int fht_double(double *buf, int log_n);
int fht_float_oop(float *in, float *out, int log_n);
int fht_double_oop(double *in, double *out, int log_n);

// In gram_scmidt.c
void CGS(int n, int m, double *X, double *Q, double *R);
void MGS(int n, int m, double *X, double *Q, double *R);
void CGS2(int n, int m, double *X, double *Q, double *R);
void RGS(int n, int m, int k, double *X, dSRHT *srht, double *Q, double *R, double *S, bool blas2, bool ffht);
void RGS_mp(int n, int m, int k, double *X, dSRHT *srht, float *Q, double *R, double *S, bool ffht);
void RGS_cb(int n, int m, int k, double *X, dSRHT *srht, float *Q, double *R, double *S, bool ffht);
void RGS_low_sync(int n, int m, int k, double *X, dSRHT *srht, double *Q, double *R, double *S, bool ffht);

// In cholesky_qr.c
void colRCholeskyQR(int n, int m, int p, int k, double *X, dSRHT *srht, double *Q, double *R, double *S);

// In block_gram_schmidt.c
void BMGS(int n, int m, int p, double *X, double *Q, double *R);
void BCGS2(int n, int m, int p, double *X, double *Q, double *R);
void BRGS(int n, int m, int p, int k, double *X, dSRHT *srht, double *Q, double *R, double *S, bool ffht);
/*void BRGS_new(int n, int m, int p, int k, double *X, dBSRHT *Bsrht, dSRHT *srht, double *Q, double *R, double *S, bool grouped);*/

// In misc.c
void cblas_sdgemv(int m, int n, double *a, float *B, double *c);
void Richardson_lsq(int k, int i, double *S, double *p, double *y);
double matrix_2norm(double *A, int n, int m);
void cblas_dscopy(int n, double *X, float *Y);
void cblas_sdcopy(int n, float *X, double *Y);

#endif // RAND_NLA_H
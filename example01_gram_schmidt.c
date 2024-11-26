/*
 * Copyright (c) 2024 Nicolas Venkovic
 * 
 * This file is part of c-rand-nla, currently a private repository.
 * 
 * This file is licensed under the MIT License.
 * For the full license text, see the LICENSE file in the root directory of this project.
 */

#include "rand_nla.h"

int main(int argc, char *argv[]) {

  int n, m, P;
  printf("Enter dimension, n: ");
  if(scanf("%d", &n)){};
  printf("Enter number of vectors, m: ");
  if(scanf("%d", &m)){};
  printf("Enter the number of processes, P: ");
  if(scanf("%d", &P)){};

  // Generate random vectors
  // X contains m n-dimensional vectors stored by row in an unrolled row major array
  double *X = (double*)mkl_malloc(n * m * sizeof(double), sizeof(double));
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      X[i * n + j] = rand() / (double) RAND_MAX;
    }
  }

  // Set the number of processes
  omp_set_num_threads(P);
  
  // Set sketching dimension
  int k = ceil(2 * m * log(n) / log(m));
  if (k > n)
    k = m;

  // Initialize SRHT
  dSRHT dsrht = SetdSrht(n, k);

  // Allocate data structures for QR decompositions
  double *Q = (double*)mkl_malloc(n * m * sizeof(double), sizeof(double));
  float *Q_sp = (float*)mkl_malloc(n * m * sizeof(float), sizeof(float));
  double *R = (double*)mkl_malloc(m * m * sizeof(double), sizeof(double));
  double *S = (double*)mkl_malloc(k * m * sizeof(double), sizeof(double));
  double *QtQ = (double*)mkl_malloc(m * m * sizeof(double), sizeof(double));
  double *QR = (double*)mkl_malloc(n * m * sizeof(double), sizeof(double));
  double dQtQ, dX;

  long seconds, microseconds;
  double dt;
  struct timeval begin, end;

  // MGS
  //
  gettimeofday(&begin, 0);
  MGS(n, m, X, Q, R);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;  
  dt = seconds + microseconds * 1e-6;
  printf("\nTime elapsed for MGS: %f sec.\n", dt);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, n, 1., Q, n, Q, n, 0., QtQ, m);
  for (int i=0; i<m; i++)
    QtQ[i * m + i] -= 1.;
  dQtQ = matrix_2norm(QtQ, m, m);
  printf("||I - Q^TQ||_2 = %g\n", dQtQ);
  
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m, 1., Q, n, R, m, 0., QR, n);
  cblas_daxpy(n * m, -1., X, 1, QR, 1);
  dX = matrix_2norm(QR, n, m);
  printf("||X - QR||_2 = %g\n", dX);

  // CGS2
  //
  gettimeofday(&begin, 0);
  CGS2(n, m, X, Q, R);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;  
  dt = seconds + microseconds * 1e-6;
  printf("\nTime elapsed for CGS2: %f sec.\n", dt);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, n, 1., Q, n, Q, n, 0., QtQ, m);
  for (int i=0; i<m; i++)
    QtQ[i * m + i] -= 1.;
  dQtQ = matrix_2norm(QtQ, m, m);
  printf("||I - Q^TQ||_2 = %g\n", dQtQ);
  
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m, 1., Q, n, R, m, 0., QR, n);
  cblas_daxpy(n * m, -1., X, 1, QR, 1);
  dX = matrix_2norm(QR, n, m);
  printf("||X - QR||_2 = %g\n", dX);

  // RGS
  //
  printf("\nSketching dimension, k: %d\n", k);
  gettimeofday(&begin, 0);
  RGS(n, m, k, X, &dsrht, Q, R, S, true, false);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;  
  dt = seconds + microseconds * 1e-6;
  printf("Time elapsed for RGS: %f sec.\n", dt);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, k, 1., S, k, S, k, 0., QtQ, m);
  for (int i=0; i<m; i++)
    QtQ[i * m + i] -= 1.;
  dQtQ = matrix_2norm(QtQ, m, m);
  printf("||I - (Theta * Q)^T(Theta * Q)||_2 = %g\n", dQtQ);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, n, 1., Q, n, Q, n, 0., QtQ, m);
  for (int i=0; i<m; i++)
    QtQ[i * m + i] -= 1.;
  dQtQ = matrix_2norm(QtQ, m, m);
  printf("||I - Q^TQ||_2 = %g\n", dQtQ);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m, 1., Q, n, R, m, 0., QR, n);
  cblas_daxpy(n * m, -1., X, 1, QR, 1);
  dX = matrix_2norm(QR, n, m);
  printf("||X - QR||_2 = %g\n", dX);

  // RGS_ffht
  //
  printf("\nSketching dimension, k: %d\n", k);
  gettimeofday(&begin, 0);
  RGS(n, m, k, X, &dsrht, Q, R, S, true, true);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;  
  dt = seconds + microseconds * 1e-6;
  printf("Time elapsed for RGS_ffht: %f sec.\n", dt);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, k, 1., S, k, S, k, 0., QtQ, m);
  for (int i=0; i<m; i++)
    QtQ[i * m + i] -= 1.;
  dQtQ = matrix_2norm(QtQ, m, m);
  printf("||I - (Theta * Q)^T(Theta * Q)||_2 = %g\n", dQtQ);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, n, 1., Q, n, Q, n, 0., QtQ, m);
  for (int i=0; i<m; i++)
    QtQ[i * m + i] -= 1.;
  dQtQ = matrix_2norm(QtQ, m, m);
  printf("||I - Q^TQ||_2 = %g\n", dQtQ);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m, 1., Q, n, R, m, 0., QR, n);
  cblas_daxpy(n * m, -1., X, 1, QR, 1);
  dX = matrix_2norm(QR, n, m);
  printf("||X - QR||_2 = %g\n", dX);

  // RGS_mp_ffht
  //
  printf("\nSketching dimension, k: %d\n", k);
  gettimeofday(&begin, 0);
  RGS_mp(n, m, k, X, &dsrht, Q_sp, R, S, true);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;  
  dt = seconds + microseconds * 1e-6;
  printf("Time elapsed for RGS_mp_ffht: %f sec.\n", dt);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, k, 1., S, k, S, k, 0., QtQ, m);
  for (int i=0; i<m; i++)
    QtQ[i * m + i] -= 1.;
  dQtQ = matrix_2norm(QtQ, m, m);
  printf("||I - (Theta * Q)^T(Theta * Q)||_2 = %g\n", dQtQ);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, n, 1., Q, n, Q, n, 0., QtQ, m);
  for (int i=0; i<m; i++)
    QtQ[i * m + i] -= 1.;
  dQtQ = matrix_2norm(QtQ, m, m);
  printf("||I - Q^TQ||_2 = %g\n", dQtQ);

  cblas_sdcopy(n * m, Q_sp, Q);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m, 1., Q, n, R, m, 0., QR, n);
  cblas_daxpy(n * m, -1., X, 1, QR, 1);
  dX = matrix_2norm(QR, n, m);
  printf("||X - QR||_2 = %g\n", dX);

  // RGS_cb_ffht
  //
  printf("\nSketching dimension, k: %d\n", k);
  gettimeofday(&begin, 0);
  RGS_cb(n, m, k, X, &dsrht, Q_sp, R, S, true);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;  
  dt = seconds + microseconds * 1e-6;
  printf("Time elapsed for RGS_cb_ffht: %f sec.\n", dt);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, k, 1., S, k, S, k, 0., QtQ, m);
  for (int i=0; i<m; i++)
    QtQ[i * m + i] -= 1.;
  dQtQ = matrix_2norm(QtQ, m, m);
  printf("||I - (Theta * Q)^T(Theta * Q)||_2 = %g\n", dQtQ);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, n, 1., Q, n, Q, n, 0., QtQ, m);
  for (int i=0; i<m; i++)
    QtQ[i * m + i] -= 1.;
  dQtQ = matrix_2norm(QtQ, m, m);
  printf("||I - Q^TQ||_2 = %g\n", dQtQ);

  cblas_sdcopy(n * m, Q_sp, Q);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m, 1., Q, n, R, m, 0., QR, n);
  cblas_daxpy(n * m, -1., X, 1, QR, 1);
  dX = matrix_2norm(QR, n, m);
  printf("||X - QR||_2 = %g\n", dX);

  /*
  // colRCholeskyQR
  //
  printf("Sketching dimension, k: %d\n", k);
  gettimeofday(&begin, 0);
  colRCholeskyQR(n, m, p, k, X, &dsrht, Q, R, S);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;  
  dt = seconds + microseconds * 1e-6;
  printf("Time elapsed for colRCholeskyQR: %f sec.\n", dt);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, k, 1., S, k, S, k, 0., QtQ, m);
  for (int i=0; i<m; i++)
    QtQ[i * m + i] -= 1.;
  dQtQ = matrix_2norm(QtQ, m, m);
  printf("||I - (Theta * Q)^T(Theta * Q)||_2 = %g\n", dQtQ);
  
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m, 1., Q, n, R, m, 0., QR, n);
  cblas_daxpy(n * m, -1., X, 1, QR, 1);
  dX = matrix_2norm(QR, n, m);
  printf("||X - QR||_2 = %g\n", dX);
  */

  mkl_free(X);
  mkl_free(Q);
  mkl_free(Q_sp);
  mkl_free(R);
  mkl_free(S);
  mkl_free(QtQ);
  mkl_free(QR);

  return 0;
}

// n=1_000_000, m=10, s=10, P=1
// MGS=0.09s, CGS2=0.10s, RGS=0.24s, RGS_FFHT=0.09s, RGS_MP_FFHT=0.09s, RGS_CB_FFHT=0.09s
// n=1_000_000, m=50, s=10, P=1
// MGS=1.72s, CGS2=1.97s, RGS=1.17s, RGS_FFHT=1.11s, RGS_MP_FFHT=0.77s, RGS_CB_FFHT=1.58s
// n=1_000_000, m=100, s=10, P=1
// MGS=6.46s, CGS2=7.72s, RGS=4.67s, RGS_FFHT=3.39s, RGS_MP_FFHT=2.02s, RGS_CB_FFHT=4.07s
// n=1_000_000, m=200, s=10, P=1
// MGS=24.22s, CGS2=30.36s, RGS=13.52s, RGS_FFHT=9.95s, RGS_MP_FFHT=6.21s, RGS_CB_FFHT=14.58s
// n=1_000_000, m=300, s=10, P=1
// MGS=57.20s, CGS2=67.88s, RGS=26.58s, RGS_FFHT=21.09s, RGS_MP_FFHT=11.54s, RGS_CB_FFHT=33.29s
// n=1_000_000, m=400, s=10, P=1
// MGS=108.82s, CGS2=128.10s, RGS=40.18s, RGS_FFHT=34.93s, RGS_MP_FFHT=18.66s, RGS_CB_FFHT=45.10s
// n=1_000_000, m=500, s=10, P=1
// MGS=165.89s, CGS2=188.79s, RGS=62.48s, RGS_FFHT=53.74s, RGS_MP_FFHT=29.39s, RGS_CB_FFHT=s
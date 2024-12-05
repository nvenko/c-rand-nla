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

  bool check_orthogonality_brgs = true;

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
  int k = ceil(m / log(m));

  // Set block size
  int s = 10;
  int p = m / s;

  // Initialize SRHT
  /*dBSRHT dBsrht = SetdBSrht(n, k, s);
  dSRHT dsrht;
  dsrht.n = dBsrht.n; dsrht.N = dBsrht.N; dsrht.log2_N = dBsrht.log2_N; dsrht.k = dBsrht.k;
  dsrht.D = dBsrht.D; dsrht.perm = dBsrht.perm; dsrht.z = dBsrht.Z;*/
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

  // BMGS
  //
  gettimeofday(&begin, 0);
  BMGS(n, m, p, X, Q, R);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;  
  dt = seconds + microseconds * 1e-6;
  printf("\nTime elapsed for BMGS: %f sec.\n", dt);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, n, 1., Q, n, Q, n, 0., QtQ, m);
  for (int i=0; i<m; i++)
    QtQ[i * m + i] -= 1.;
  dQtQ = matrix_2norm(QtQ, m, m);
  printf("||I - Q^TQ||_2 = %g\n", dQtQ);
  
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m, 1., Q, n, R, m, 0., QR, n);
  cblas_daxpy(n * m, -1., X, 1, QR, 1);
  dX = matrix_2norm(QR, n, m);
  printf("||X - QR||_2 = %g\n", dX);
  
  
  // BCGS2
  //
  gettimeofday(&begin, 0);
  BCGS2(n, m, p, X, Q, R);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;  
  dt = seconds + microseconds * 1e-6;
  printf("\nTime elapsed for BCGS2: %f sec.\n", dt);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, n, 1., Q, n, Q, n, 0., QtQ, m);
  for (int i=0; i<m; i++)
    QtQ[i * m + i] -= 1.;
  dQtQ = matrix_2norm(QtQ, m, m);
  printf("||I - Q^TQ||_2 = %g\n", dQtQ);
  
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m, 1., Q, n, R, m, 0., QR, n);
  cblas_daxpy(n * m, -1., X, 1, QR, 1);
  dX = matrix_2norm(QR, n, m);
  printf("||X - QR||_2 = %g\n", dX);
  

  // BRGS
  //
  printf("\nSketching dimension, k: %d\n", k);
  gettimeofday(&begin, 0);
  BRGS(n, m, p, k, X, &dsrht, Q, R, S, false);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;  
  dt = seconds + microseconds * 1e-6;
  printf("Time elapsed for BRGS: %f sec.\n", dt);

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

  // BRGS_ffht
  //
  printf("\nSketching dimension, k: %d\n", k);
  gettimeofday(&begin, 0);
  BRGS(n, m, p, k, X, &dsrht, Q, R, S, true);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;  
  dt = seconds + microseconds * 1e-6;
  printf("Time elapsed for BRGS_ffht: %f sec.\n", dt);

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

  // BRGS_new
  //
  /*printf("Sketching dimension, k: %d\n", k);
  gettimeofday(&begin, 0);
  BRGS_new(n, m, p, k, X, &dBsrht, &dsrht, Q, R, S, true);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;  
  dt = seconds + microseconds * 1e-6;
  printf("Time elapsed for BRGS_new: %f sec.\n", dt);

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
  printf("||X - QR||_2 = %g\n", dX);*/

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
// BMGS=0.07s, BCGS2=0.07s, BRGS=0.23s, BRGS_FFHT=0.08s
// n=1_000_000, m=50, s=10, P=1
// BMGS=0.64s, BCGS2=0.86s, BRGS=2.04s, BRGS_FFHT=0.59s
// n=1_000_000, m=100, s=10, P=1
// BMGS=1.62s, BCGS2=2.25s, BRGS=3.92s, BRGS_FFHT=1.57s
// n=1_000_000, m=200, s=10, P=1
// BMGS=4.90s, BCGS2=8.43s, BRGS=9.93s, BRGS_FFHT=4.31s
// n=1_000_000, m=300, s=10, P=1
// BMGS=10.73s, BCGS2=18.65s, BRGS=17.17s, BRGS_FFHT=8.56s
// n=1_000_000, m=400, s=10, P=1
// BMGS=19.41s, BCGS2=32.83s, BRGS=23.75s, BRGS_FFHT=13.61s
// n=1_000_000, m=500, s=10, P=1
// BMGS=28.99s, BCGS2=51.10s, BRGS=33.74s, BRGS_FFHT=20.43s
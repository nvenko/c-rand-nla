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

  int s = 10;

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
  printf("\nSketching dimension, k: %d\n", k);

  // Initialize SRHT
  dSRHT dsrht = SetdSrht(n, k);

  long seconds, microseconds;
  double dt;
  struct timeval begin, end;

  double *Theta_X1 = (double*)mkl_malloc(s * k * sizeof(double), sizeof(double));
  double *Theta_X2 = (double*)mkl_malloc(s * k * sizeof(double), sizeof(double));
  //
  gettimeofday(&begin, 0);
  BlockMatrixFreeTheta(X, &dsrht, s, Theta_X1);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;  
  dt = seconds + microseconds * 1e-6;
  printf("Time elapsed for dMatrixFreeTheta: %f sec.\n", dt);
  //
  gettimeofday(&begin, 0);
  BlockMatrixFreeTheta_ffht(X, &dsrht, s, Theta_X2);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;  
  dt = seconds + microseconds * 1e-6;
  printf("Time elapsed for dMatrixFreeTheta_ffht: %f sec.\n", dt);
  //
  for (int i=0; i<s*k; i++) Theta_X1[i] -= Theta_X2[i];
  double norm2 = cblas_dnrm2(s * k, Theta_X1, 1);
  printf("||Theta_x1 - Theta_x2||_2 = %g\n", norm2);
  //
  mkl_free(Theta_X1);
  mkl_free(Theta_X2);
  mkl_free(X);

  return 0;
}
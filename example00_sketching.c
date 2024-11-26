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
  printf("\nSketching dimension, k: %d\n", k);

  // Initialize SRHT
  dSRHT dsrht = SetdSrht(n, k);

  long seconds, microseconds;
  double dt;
  struct timeval begin, end;

  double *Theta_x1 = (double*)mkl_malloc(k * sizeof(double), sizeof(double));
  double *Theta_x2 = (double*)mkl_malloc(k * sizeof(double), sizeof(double));
  //
  gettimeofday(&begin, 0);
  dMatrixFreeTheta(X, &dsrht, Theta_x1);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;  
  dt = seconds + microseconds * 1e-6;
  printf("Time elapsed for dMatrixFreeTheta: %f sec.\n", dt);
  //
  gettimeofday(&begin, 0);
  dMatrixFreeTheta_ffht(X, &dsrht, Theta_x2);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;  
  dt = seconds + microseconds * 1e-6;
  printf("Time elapsed for dMatrixFreeTheta_ffht: %f sec.\n", dt);
  //
  for (int i=0; i<k; i++) Theta_x1[i] -= Theta_x2[i];
  double norm2 = cblas_dnrm2(k, Theta_x1, 1);
  printf("||Theta_x1 - Theta_x2||_2 = %g\n", norm2);
  //
  mkl_free(Theta_x1);
  mkl_free(Theta_x2);
  mkl_free(X);

  return 0;
}
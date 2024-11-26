/*
 * Copyright (c) 2024 Nicolas Venkovic
 * 
 * This file is part of c-rand-nla, currently a private repository.
 * 
 * This file is licensed under the MIT License.
 * For the full license text, see the LICENSE file in the root directory of this project.
 */

#include "rand_nla.h"

void dumb_fht(double *buf, int log_n) {
    int n = 1 << log_n;
    for (int i = 0; i < log_n; ++i) {
        int s1 = 1 << i;
        int s2 = s1 << 1;
        for (int j = 0; j < n; j += s2) {
            for (int k = 0; k < s1; ++k) {
                double u = buf[j + k];
                double v = buf[j + k + s1];
                buf[j + k] = u + v;
                buf[j + k + s1] = u - v;
            }
        }
    }
}

/*
dSRHT SetdSrht(int n, int k)
Sets the subsampled randomized Hadamard transform structure.

In:
  - int n: Dimension of the vectors.
  - int k: Sketching dimension.

Out:
  - dSRHT srht: Subsampled randomized Hadamard transform structure.
*/
dSRHT SetdSrht(int n, int k){
  dSRHT srht;
  srht.n = n;
  srht.log2_N = (int) ceil(log(n) / log(2));
  srht.N = (int) pow(2, srht.log2_N);
  srht.k = k;
  int *D;
  D = (int*)mkl_malloc(n * sizeof(int), sizeof(int));
  double rho;
  for (int i=0; i<n; i++) {
    rho = rand() / (double) RAND_MAX;
    if (rho < 0.5) {
      D[i] = 1;
    }
    else {
      D[i] = -1;
    }
  }
  srht.D = D;
  int *perm = RandomPermutation(srht.N);
  srht.perm = (int*)mkl_malloc(k * sizeof(int), sizeof(int));
  for (int i=0; i<k; i++)
    srht.perm[i] = perm[i];
  mkl_free(perm);
  srht.z = (double*)mkl_malloc(srht.N * sizeof(double), sizeof(double));
  return srht;
}

/*
dBSRHT SetdBSrht(int n, int k, int s)
Sets the block subsampled randomized Hadamard transform structure.

In:
  - int n: Dimension of the vectors.
  - int k: Sketching dimension.
  - int s: Block size.
Out:
  - dBSRHT srht: Block subsampled randomized Hadamard transform structure.
*/
/*dBSRHT SetdBSrht(int n, int k, int s){
  dBSRHT srht;
  srht.n = n;
  srht.log2_N = (int) ceil(log(n) / log(2));
  srht.N = (int) pow(2, srht.log2_N);
  srht.k = k;
  srht.s = s;
  int *D;
  D = (int*)mkl_malloc(n * sizeof(int), sizeof(int));
  double rho;
  for (int i=0; i<n; i++) {
    rho = rand() / (double) RAND_MAX;
    if (rho < 0.5) {
      D[i] = 1;
    }
    else {
      D[i] = -1;
    }
  }
  srht.D = D;
  int *perm = RandomPermutation(srht.N);
  srht.perm = (int*)mkl_malloc(k * sizeof(int), sizeof(int));
  for (int i=0; i<k; i++)
    srht.perm[i] = perm[i];
  mkl_free(perm);
  srht.Z = (double*)mkl_malloc(srht.s * srht.N * sizeof(double), sizeof(double));
  return srht;
}*/

/*
sSRHT SetsSrht(int n, int k)
Sets the subsampled randomized Hadamard transform structure.

In:
  - int n: Dimension of the vectors.
  - int k: Sketching dimension.

Out:
  - sSRHT srht: Subsampled randomized Hadamard transform structure.
*/
sSRHT SetsSrht(int n, int k){
  sSRHT srht;
  srht.n = n;
  srht.log2_N = (int) ceil(log(n) / log(2));
  srht.N = (int) pow(2, srht.log2_N);
  srht.k = k;
  int *D;
  D = (int*)mkl_malloc(n * sizeof(int), sizeof(int));
  double rho;
  for (int i=0; i<n; i++) {
    rho = rand() / (double) RAND_MAX;
    if (rho < 0.5) {
      D[i] = 1;
    }
    else {
      D[i] = -1;
    }
  }
  srht.D = D;
  int *perm = RandomPermutation(srht.N);
  srht.perm = (int*)mkl_malloc(k * sizeof(int), sizeof(int));
  for (int i=0; i<k; i++)
    srht.perm[i] = perm[i];
  mkl_free(perm);
  srht.z = (float*)mkl_malloc(srht.N * sizeof(float), sizeof(float));
  return srht;
}

/*
int *RandomPermutation(int n)
Returns an vector of randomly permuted indices.

In:
  - int n: Number of indices.

Out:
  - int *perm: Vector of randomly permuted indices.
*/
int *RandomPermutation(int n) {
  int *perm = (int*)mkl_malloc(n * sizeof(int), sizeof(int));
  for (int i=0; i<n; i++)
    perm[i] = i;
  for (int i=0; i<n; i++) {
    int j, t;
    j = rand() % (n-i) + i;
    t = perm[j]; perm[j] = perm[i]; perm[i] = t;
  }
  return perm;
}

/*
void dfwht(double *x, int n, int *D, double *z, int N, int log2_N)
Return the fast Walsh Hadamard transform of a vector.

In:
  - double *x: n-dimensional vector.
  - int n: Dimension of the vector.
  - int *D: Sign flips.
  - double *z: N-dimensional padded-transformed vector.
  - int N: Smallest power of 2 greater or equal to n.
  - int log2_N: log2 of N.

*/
void dfwht(double *x, int n, int *D, double *z, int N, int log2_N) {
  int h = 1;
  int i;
  #pragma omp parallel for shared(z, x, D, n) private(i)
  for (i=0; i<n; i++) z[i] = D[i] * x[i];
  #pragma omp parallel for shared(z, N) private(i)
  for (i=n; i<N; i++) z[i] = 0.;
  for (int _=0; _<log2_N; _++) {
    int dh = 2 * h;
    #pragma omp parallel for shared(z, h, N) private(i)
    for (int i=0; i<N; i+=dh) {
      for (int j=i; j<i+h; j++) {
        double w = z[j];
        double y = z[j + h];
        z[j] = w + y;
        z[j + h] = w - y;
      }
    }
    h *= 2;
  }
}

/*void Blockdfwht(double *X, int n, int *D, double *Z, int s, int N, int log2_N) {
  for (int i=0; i<n; i++) {
    int d = D[i];
    double *z = Z + i * s;
    double *x = X + i;
    for (int r=0; r<s; r++) z[r] = d * x[r * n];
  }
  for (int i=n; i<N; i++) {
    double *z = Z + i * s;
    for (int r=0; r<s; r++) z[r] = 0.;
  }
  int h = 1;
  double *w = (double*)mkl_malloc(s * sizeof(double), sizeof(double));
  double *y = (double*)mkl_malloc(s * sizeof(double), sizeof(double));
  for (int _=0; _<log2_N; _++) {
    int dh = 2 * h;
    for (int i=0; i<N; i+=dh) {
      for (int j=i; j<i+h-1; j++) {
        double *z1 = Z + j * s; 
        double *z2 = z1 + h * s;
        for (int r=0; r<s; r++) {
          w[r] = z1[r];
          y[r] = z2[r];
        }
        for (int r=0; r<s; r++) {
          z1[r] = w[r] + y[r];
          z2[r] = w[r] - y[r];
        }
      }
    }
    h *= 2;
  }
  mkl_free(w);
  mkl_free(y);
}*/

/*void Blockdfwht(double *X, int n, int *D, double *Z, int s, int N, int log2_N) {
  for (int r=0; r<s; r++) {
    double *z = Z + r * N;
    double *x = X + r * n;
    for (int i=0; i<n; i++) z[i] = D[i] * x[i];
    for (int i=n; i<N; i++) z[i] = 0.;
  }
  for (int r=0; r<s; r++) {
    int h = 1;
    double *z = Z + r * N;
    for (int _=0; _<log2_N; _++) {
      int dh = 2 * h;
      for (int i=0; i<N; i+=dh) {
        for (int j=i; j<i+h-1; j++) {
          double w = z[j];
          double y = z[j + h];
          z[j] = w + y;
          z[j + h] = w - y;
        }
      }
      h *= 2;
    }
  }
}*/

/*
void sfwht(float *a, int n, float *z, int N, int log2_N)
Return the fast Walsh Hadamard transform of a vector.

In:
  - float *x: n-dimensional vector.
  - int n: Dimension of the vector.
  - int *D: Sign flips.
  - float *z: N-dimensional padded-transformed vector.
  - int N: Smallest power of 2 greater or equal to n.
  - int log2_N: log2 of N.
*/
void sfwht(float *x, int n, int *D, float *z, int N, int log2_N) {
  int h = 1;
  int i;
  #pragma omp parallel for shared(z, x, D, n) private(i)
  for (i=0; i<n; i++) z[i] = D[i] * x[i];
  #pragma omp parallel for shared(z, N) private(i)
  for (i=n; i<N; i++) z[i] = 0.;
  for (int r=0; r<log2_N; r++) {
    int dh = 2 *h;
    #pragma omp parallel for shared(z, h, N) private(i)
    for (int i=0; i<N; i+=dh) {
      for (int j=i; j<i+h-1; j++) {
        float w = z[j];
        float y = z[j + h];
        z[j] = w + y;
        z[j + h] = w - y;
      }
    }
    h *= 2;
  }
}

/*
void dMatrixFreeTheta(double *x, dSRHT *srht, double *Theta_x)
Applies matrix-free random sketching.

In:
  - double *x: n-dimensional vector.
  - dSRHT *srht: Subsampled randomized Hadamard transform structure.
  - double *Theta_x: Randomly sketched k-dimensional vector.
*/
void dMatrixFreeTheta(double *x, dSRHT *srht, double *Theta_x) {
  int i;
  double fac = 1. / sqrt(srht->k);
  dfwht(x, srht->n, srht->D, srht->z, srht->N, srht->log2_N);
  #pragma omp parallel for shared(Theta_x, fac, srht->z, srht->perm) private(i)
  for (i=0; i<srht->k; i++) Theta_x[i] = fac * srht->z[srht->perm[i]];
}

/*
void dMatrixFreeTheta_ffht(double *x, dSRHT *srht, double *Theta_x)
Applies matrix-free random sketching using FFHT routine.

In:
  - double *x: n-dimensional vector.
  - dSRHT *srht: Subsampled randomized Hadamard transform structure.
  - double *Theta_x: Randomly sketched k-dimensional vector.
*/
void dMatrixFreeTheta_ffht(double *x, dSRHT *srht, double *Theta_x) {
  int i;
  double fac = 1. / sqrt(srht->k);
  #pragma omp parallel for shared(srht->z, srht->D, x) private(i)
  for (i=0; i<srht->n; i++) srht->z[i] = srht->D[i] * x[i];
  #pragma omp parallel for shared(srht->z) private(i)
  for (i=srht->n; i<srht->N; i++) srht->z[i] = 0.;
  fht_double(srht->z, srht->log2_N);
  //dumb_fht(srht->z, srht->log2_N);
  #pragma omp parallel for shared(Theta_x, fac, srht->z, srht->perm) private(i)
  for (i=0; i<srht->k; i++) Theta_x[i] = fac * srht->z[srht->perm[i]];
}

/*
void sMatrixFreeTheta(float *x, sSRHT *srht, float *Theta_x)
Applies matrix-free random sketching with mixed precision data.

In:
  - float *x: n-dimensional vector.
  - sSRHT *srht: Subsampled randomized Hadamard transform structure.
  - float *Theta_x: Randomly sketched k-dimensional vector.
*/
void sMatrixFreeTheta(float *x, sSRHT *srht, float *Theta_x) {
  int i;
  double fac = 1. / sqrt(srht->k);
  sfwht(x, srht->n, srht->D, srht->z, srht->N, srht->log2_N);
  #pragma omp parallel for shared(Theta_x, fac, srht->z, srht->perm) private(i)
  for (i=0; i<srht->k; i++) Theta_x[i] = fac * srht->z[srht->perm[i]];
}

/*
void sMatrixFreeTheta_ffht(float *x, sSRHT *srht, float *Theta_x)
Applies matrix-free random sketching using FFHT routine.

In:
  - float *x: n-dimensional vector.
  - sSRHT *srht: Subsampled randomized Hadamard transform structure.
  - float *Theta_x: Randomly sketched k-dimensional vector.
*/
void sMatrixFreeTheta_ffht(float *x, sSRHT *srht, float *Theta_x) {
  int i;
  double fac = 1. / sqrt(srht->k);x[i];
  #pragma omp parallel for shared(srht->z, srht->D, x) private(i)
  for (i=0; i<srht->n; i++) srht->z[i] = srht->D[i] * x[i];
  #pragma omp parallel for shared(srht->z) private(i)
  for (i=srht->n; i<srht->N; i++) srht->z[i] = 0.;
  fht_float(srht->z, srht->log2_N);
  #pragma omp parallel for shared(Theta_x, srht->z, srht->perm) private(i)
  for (i=0; i<srht->k; i++) Theta_x[i] = fac * srht->z[srht->perm[i]];
}

/*
void BlockMatrixFreeTheta(double *X, dSRHT *srht, int s, double *Theta_X)
Applies matrix-free random sketching.

In:
  - double *X: block of s n-dimensional vector stored by column in a column major array.
  - dSRHT *srht: Subsampled randomized Hadamard transform structure.
  - int s: Number of vectors per block.
  - double *Theta_X: s randomly sketched k-dimensional vectors stored by column in a column major array.
*/
void BlockMatrixFreeTheta(double *X, dSRHT *srht, int s, double *Theta_X) {
  int i, j;
  double fac = 1. / sqrt(srht->k);
  #pragma omp parallel for private(j)
  for (j=0; j<s; j++) {
    dfwht(X + j * srht->n, srht->n, srht->D, srht->z, srht->N, srht->log2_N);
    #pragma omp parallel for shared(Theta_X, z) private(i)
    for (i=0; i<srht->k; i++) Theta_X[srht->k * j + i] = fac * srht->z[srht->perm[i]];
  }
}

/*void BlockMatrixFreeTheta_grouped(double *X, dBSRHT *srht, double *Theta_X) {
  Blockdfwht(X, srht->n, srht->D, srht->Z, srht->s, srht->N, srht->log2_N);
  double fac = 1. / sqrt(srht->k);
  for (int i=0; i<srht->k; i++) {
    double *z = srht->Z + srht->perm[i] * srht->s;
    double *Theta_x = Theta_X + i;
    for (int r=0; r<srht->s; r++) Theta_x[srht->k * r] = fac * z[r];
  }
}*/

/*void BlockMatrixFreeTheta_grouped(double *X, dBSRHT *srht, double *Theta_X) {
  Blockdfwht(X, srht->n, srht->D, srht->Z, srht->s, srht->N, srht->log2_N);
  double fac = 1. / sqrt(srht->k);
  for (int r=0; r<srht->s; r++) {
    double *z = srht->Z + srht->N * r;
    double *Theta_x = Theta_X + srht->k * r;
    for (int i=0; i<srht->k; i++) Theta_x[i] = fac * z[srht->perm[i]];
  }
}*/

/*
void BlockMatrixFreeTheta_ffht(double *X, dSRHT *srht, int s, double *Theta_X)
Applies matrix-free random sketching using FFHT.

In:
  - double *X: block of s n-dimensional vector stored by column in a column major array.
  - dSRHT *srht: Subsampled randomized Hadamard transform structure.
  - int s: Number of vectors per block.
  - double *Theta_X: s randomly sketched k-dimensional vectors stored by column in a column major array.
*/
void BlockMatrixFreeTheta_ffht(double *X, dSRHT *srht, int s, double *Theta_X) {
  int i, j;
  double fac = 1. / sqrt(srht->k);
  #pragma omp parallel for private(j)
  for (j=0; j<s; j++) {
    for (i=0; i<srht->n; i++) srht->z[i] = srht->D[i] * X[j * srht->n + i];
    for (i=srht->n; i<srht->N; i++) srht->z[i] = 0.;
    fht_double(srht->z, srht->log2_N);
    //dumb_fht(srht->z, srht->log2_N);
    #pragma omp parallel for shared(Theta_X, z) private(i)
    for (i=0; i<srht->k; i++) Theta_X[srht->k * j + i] = fac * srht->z[srht->perm[i]];
  }
}
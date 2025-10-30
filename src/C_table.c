#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// Generate one standard normal sample using Box–Muller
double randn_one() {
    double u1 = ((double) rand() + 1.0) / ((double) RAND_MAX + 2.0);
    double u2 = ((double) rand() + 1.0) / ((double) RAND_MAX + 2.0);

    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// Generate an array of size m with N(0,1) samples
void randn(double *out, int m) {
    for (int i = 0; i < m; i++) {
        out[i] = randn_one();
    }
}

double* C_table(int V, int m) {
    srand(time(NULL));  // seed RNG

    double *arr = malloc(m * sizeof(double));

    randn(arr, m);

    return arr;
}
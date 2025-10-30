#include <stdio.h>
#include <stdlib.h>

#include "dot_product.h"

// Matrix multiplication using dot product
// A: (rowsA x colsA), B: (colsA x colsB), C: (rowsA x colsB)
double* matmul(double *A, double *B, int rowsA, int colsA, int colsB) {
    double *C = malloc(rowsA * colsB * sizeof(double));

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {

            // build temporary column vector from B
            double colB[colsA];

            for (int k = 0; k < colsA; k++) {
                colB[k] = B[k * colsB + j];
            }

            // compute dot product between row of A and column of B
            double *rowA = &A[i * colsA];

            C[i * colsB + j] = dot_product(colsA, rowA, colB);
        }
    }

    return C;
}
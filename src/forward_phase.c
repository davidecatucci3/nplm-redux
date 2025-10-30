// built-in files
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

// external lfiles
#include "C_table.h"
#include "dot_product.h"
#include "mat_mul.h"

int main() {
    // hyperparameters
    int V = 16384;
    int m = 60;
    int h = 50;
    int n = 5;

    // varaibles 
    int ids[] = {1, 434, 45, 123, 3333}; // token ids vector
    double* C = C_table(V, m);
    double x[n][m];
    double x_flat[n * m];

    // perform forward computation for the word features layer
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        
        for (int j = 0; j < m; j++) {
            x[i][j] = C[id * m + j];  
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            x_flat[i * m + j] = x[i][j];
        }
    }

    // perform forward computation for the hidden layer
    double* H = C_table(V, m);
    double* o = matmul(H, x_flat, h, n*m, h);

    for (int i = 0; i < h; i++) {
        o[i] = tanh(o[i]);
    }

    // perform forward computation for output units in the i-th block

    return 0;
}


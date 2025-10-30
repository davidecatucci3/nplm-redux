// built-in files
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
//#include <mpi.h>

// external files
#include "embedding_matrix.h"
#include "dot_product.h"
#include "mat_mul.h"

int main() {
    // hyperparameters
    int V = 16384;
    int m = 60;
    int h = 50;
    int n = 5;

    // main variables 
    int ids[] = {1, 434, 45, 123, 3333}; // token ids vector
    double* C = embedding_matrix(V, m); // C embedding matrix
    double x[n][m]; 
    double x_flat[n * m]; //input vector neural network that has been flattened 

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
    double* H = embedding_matrix(h, n*m); // weights first layer
    double d[h]; // bias first layer

    for (int i = 0; i < h; i++) {
        d[i] = i;

    }

    double* o = matmul(H, x_flat, h, n*m, h); // output first layer

    for (int i = 0; i < h; i++) {
        o[i] = tanh(o[i] + d[i]);
    }

    // perform forward computation for output units in the i-th block

    return 0;
}


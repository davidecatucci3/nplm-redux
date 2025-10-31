// built-in files
#include <stdbool.h>
#include <stdlib.h>
#include <cblas.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

// external files
#include "embedding_matrix.h"

int main() {
    // seed
    srand(time(NULL));

    // hyperparameters
    int V = 512;
    int m = 32;
    int h = 16;
    int n = 2;

    // main variables 
    int ids[] = {1, 434, 45, 123, 3333}; // token ids vector
    int next_id = 23;
    double* C = embedding_matrix(V, m); // C embedding matrix 

    // perform forward computation for the word features layer
    double* x_flat = malloc(n * m * sizeof(double)); //input vector neural network that has been flattened 

    for (int i = 0; i < n; i++) {
        int id = ids[i];
        
        for (int j = 0; j < m; j++) {
            x_flat[i * m + j] = C[id * m + j];  
        }
    }

    // perform forward computation for the hidden layer
    double* H = embedding_matrix(h, n*m); // weights first layer
    double* d = malloc(h * sizeof(double)); // bias first layer
    double* o = malloc(h * sizeof(double)); // output vector first layer

    cblas_dgemv( // BLAS faster matrix mul
        CblasRowMajor,     // row-major layout
        CblasNoTrans,      // don't transpose A
        h, n*m,
        1.0,
        H, n*m,           
        x_flat, 1,         
        0.0,
        o, 1               
    );

    for (int i = 0; i < h; i++) {
        o[i] = tanh(o[i] + d[i]);
    }

    // perform forward computation for output units in the i-th block
    int rank, comm_sz;
    int M = 8;
    double S = 0;
    double local_s = 0;
    double* local_U = embedding_matrix(V / M, h);
    double* local_b = malloc((V/M) * sizeof(double));
    double* local_y = malloc((V/M) * sizeof(double));
    double* local_p = malloc((V/M) * sizeof(double));
    double* p = malloc(V * sizeof(double));

    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    cblas_dgemv( // BLAS faster matrix mul
        CblasRowMajor,     // row-major layout
        CblasNoTrans,      // don't transpose A
        V/M, h,
        1.0,
        local_U, h,           
        o, 1,         
        0.0,
        local_y, 1               
    );

    for (int i = 0; i < V/M; i++) {
        local_y[i] = tanh(local_y[i] + local_b[i]);
        local_p[i] = exp(local_y[i]);
            
        local_s += local_p[i];
    } 

    // compute and share S among the processors
    MPI_Allreduce(&local_s, &S, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // normalize the probabilities
    for (int i = 0; i < V/M; i++) {
        local_p[i] /= S;
    }

    MPI_Allgather(local_p, V/M, MPI_DOUBLE, p, V/M, MPI_DOUBLE, MPI_COMM_WORLD);

    if (rank == 0) {
        double L = log(p[next_id]);

        printf("Loss: %lf, %lf, %lf", L, p[next_id], S);
    }

    MPI_Finalize();

    return 0;
}


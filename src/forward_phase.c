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

int forward() {
    return 0;
}

int main() {
    // seed
    srand(time(NULL));

    // hyperparameters
    int V = 512;
    int m = 32;
    int h = 16;
    int n = 2;

    // main variables 
    int ids[] = {23, 34}; // w1 and w2
    int next_id = 12; // w3
    double* C = embedding_matrix(V, m); // C embedding matrix 
    int* vocab = malloc(V * sizeof(int)); // C embedding matrix 

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
        CblasRowMajor,    
        CblasNoTrans,    
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
    MPI_Init(NULL, NULL);

    int rank, comm_sz;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    double S = 0;
    double local_s = 0;
    double* local_U = embedding_matrix(V / comm_sz, h); // weights second layer
    double* local_b = malloc((V/comm_sz) * sizeof(double)); // bias second layer
    double* local_y = malloc((V/comm_sz) * sizeof(double));
    double* local_p = malloc((V/comm_sz) * sizeof(double));
    double* p = malloc(V * sizeof(double));

    cblas_dgemv( // BLAS faster matrix mul
        CblasRowMajor,     
        CblasNoTrans,      
        V/comm_sz, h,
        1.0,
        local_U, h,           
        o, 1,         
        0.0,
        local_y, 1               
    );

    for (int i = 0; i < V/comm_sz; i++) {
        local_y[i] = local_y[i] + local_b[i];
        local_p[i] = exp(local_y[i]);
            
        local_s += local_p[i];
    } 

    // compute and share S among the processors
    MPI_Allreduce(&local_s, &S, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // normalize the probabilities
    for (int i = 0; i < V/comm_sz; i++) {
        local_p[i] /= S;
    }

    MPI_Allgather(local_p, V/comm_sz, MPI_DOUBLE, p, V/comm_sz, MPI_DOUBLE, MPI_COMM_WORLD);


    // compute loss
    if (rank == 0) {
        double L = 0; // total loss

        for (int i = 0; i < V; i++) {
            double li = log(p[vocab[i]]); // loss of wi
             
            L += li;
        }

        L = L / V;

        printf("Loss: %lf", L);
    }

    MPI_Finalize();

    return 0;
}


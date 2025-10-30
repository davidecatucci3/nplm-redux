#include <stdio.h>
#include <math.h>

double dot_product(int n, double* a, double* b) { 
    double res = 0;

    for (int i = 0; i < n; i++) {
        res += a[i]*b[i];
    } 

    return res;
}
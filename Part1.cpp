#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define MAX 10000

double A[MAX][MAX], x[MAX], y[MAX];
void initialize_arrays() {
    for (int i = 0; i < MAX; i++) {
        x[i] = (double)rand() / RAND_MAX;
        for (int j = 0; j < MAX; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
        }
    }
}
void reset_y() {
    for (int i = 0; i < MAX; i++) {
        y[i] = 0;
    }
}
double row_major_access() {
    clock_t start = clock();
    for (int i = 0; i < MAX; i++) {
        for (int j = 0; j < MAX; j++) {
            y[i] += A[i][j] * x[j];
        }
    }
    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}
double column_major_access() {
    clock_t start = clock();
    for (int j = 0; j < MAX; j++) {
        for (int i = 0; i < MAX; i++) {
            y[i] += A[i][j] * x[j];
        }
    }
    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}
int main() {
    srand(time(NULL));
    initialize_arrays();
    printf("Row-major access time: %f seconds\n", row_major_access());
    reset_y();
    printf("Column-major access time: %f seconds\n", column_major_access());
    return 0;
}
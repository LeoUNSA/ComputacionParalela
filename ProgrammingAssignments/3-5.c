#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

void Read_matrix(char* prompt, double local_A[], int local_n, int n);
void Read_vector(char* prompt, double local_x[], int local_n, int my_rank);
void Print_vector(char* title, double local_y[], int local_n, int my_rank);

int main(void) {
    int my_rank, comm_sz;
    int n, local_n;
    double *local_A = NULL;
    double *local_x = NULL;
    double *local_y = NULL;
    double *x = NULL;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // Process 0 reads n and broadcasts it
    if (my_rank == 0) {
        printf("Enter matrix order n: ");
        fflush(stdout);
        scanf("%d", &n);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_n = n / comm_sz;

    // Allocate memory
    local_A = (double*)malloc(n * local_n * sizeof(double));
    local_x = (double*)malloc(local_n * sizeof(double));
    local_y = (double*)malloc(local_n * sizeof(double));
    
    if (my_rank == 0) {
        x = (double*)malloc(n * sizeof(double));
    }

    // Read and distribute matrix
    Read_matrix("Enter the matrix", local_A, local_n, n);
    
    // Read and distribute vector
    if (my_rank == 0) {
        printf("Enter the vector:\n");
        fflush(stdout);
        for (int i = 0; i < n; i++)
            scanf("%lf", &x[i]);
    }
    MPI_Scatter(x, local_n, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local matrix-vector product
    for (int i = 0; i < local_n; i++) {
        local_y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            local_y[i] += local_A[j * local_n + i] * local_x[i];
        }
    }

    // Gather results
    double* y = NULL;
    if (my_rank == 0) {
        y = (double*)malloc(n * sizeof(double));
    }
    MPI_Gather(local_y, local_n, MPI_DOUBLE, y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print results
    if (my_rank == 0) {
        printf("The product is\n");
        for (int i = 0; i < n; i++) {
            printf("%f ", y[i]);
        }
        printf("\n");
        free(y);
        free(x);
    }

    // Free memory
    free(local_A);
    free(local_x);
    free(local_y);

    MPI_Finalize();
    return 0;
}

void Read_matrix(char* prompt, double local_A[], int local_n, int n) {
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    double* temp = NULL;
    if (my_rank == 0) {
        temp = (double*)malloc(n * n * sizeof(double));
        printf("%s\n", prompt);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                scanf("%lf", &temp[i * n + j]);
            }
        }
    }

    // Scatter the matrix by columns
    for (int i = 0; i < n; i++) {
        MPI_Scatter(&temp[i * n], local_n, MPI_DOUBLE, 
                   &local_A[i * local_n], local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (my_rank == 0) {
        free(temp);
    }
}
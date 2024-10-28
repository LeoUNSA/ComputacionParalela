#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
void Read_matrix(char* prompt, double local_A[], int local_n, int n, int my_rank);
void Read_vector(char* prompt, double x[], int n, int my_rank);
void Print_matrix(char* title, double local_A[], int local_n, int n, int my_rank);
void Print_vector(char* title, double y[], double local_y[], int n, int local_n, int my_rank);
int main(void) {
    int my_rank, comm_sz;
    double* local_A = NULL;
    double* local_x = NULL;
    double* local_y = NULL;
    double* x = NULL;  // The complete vector
    double* y = NULL;  // The complete result
    int n;            // Matrix order
    int local_n;      // Number of columns per process
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Process 0 reads matrix size
    if (my_rank == 0) {
        printf("Enter matrix order n: ");
        scanf("%d", &n);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_n = n/comm_sz;  // Number of columns per process
    // Allocate memory for local arrays
    local_A = malloc(n * local_n * sizeof(double));
    local_x = malloc(local_n * sizeof(double));
    local_y = malloc(n * sizeof(double));
    if (my_rank == 0) {
        x = malloc(n * sizeof(double));
        y = malloc(n * sizeof(double));
    }
    // Read and distribute matrix
    Read_matrix("Enter the matrix", local_A, local_n, n, my_rank);
    // Read and distribute vector
    if (my_rank == 0) {
        printf("Enter the vector:\n");
        for (int i = 0; i < n; i++)
            scanf("%lf", &x[i]);
    }
    // Scatter vector x to all processes
    MPI_Scatter(x, local_n, MPI_DOUBLE, local_x, local_n, 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local matrix-vector product
    for (int i = 0; i < n; i++) {
        local_y[i] = 0.0;
        for (int j = 0; j < local_n; j++)
            local_y[i] += local_A[i*local_n + j] * local_x[j];
    }
    // Sum up partial results using MPI_Reduce_scatter
    MPI_Reduce_scatter(local_y, y, &local_n, MPI_DOUBLE, 
                      MPI_SUM, MPI_COMM_WORLD);
    // Print results
    Print_vector("The product is", y, local_y, n, local_n, my_rank);
    // Free memory
    free(local_A);
    free(local_x);
    free(local_y);
    if (my_rank == 0) {
        free(x);
        free(y);
    }
    MPI_Finalize();
    return 0;
}
void Read_matrix(char* prompt, double local_A[], int local_n, int n, int my_rank) {
    double* A = NULL;
    int local_ok = 1;

    if (my_rank == 0) {
        A = malloc(n * n * sizeof(double));
        printf("%s\n", prompt);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                scanf("%lf", &A[i*n + j]);
        // Distribute matrix by columns to all processes
        for (int i = 0; i < n; i++)
            for (int j = 0; j < local_n; j++)
                local_A[i*local_n + j] = A[i*n + j];
        for (int dest = 1; dest < comm_sz; dest++) {
            int start_col = dest * local_n;
            for (int i = 0; i < n; i++)
                MPI_Send(&A[i*n + start_col], local_n, MPI_DOUBLE, 
                        dest, 0, MPI_COMM_WORLD);
        }
        free(A);
    } else {
        for (int i = 0; i < n; i++)
            MPI_Recv(&local_A[i*local_n], local_n, MPI_DOUBLE, 
                    0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}
void Print_vector(char* title, double y[], double local_y[], 
                 int n, int local_n, int my_rank) {
    if (my_rank == 0) {
        printf("%s\n", title);
        for (int i = 0; i < n; i++)
            printf("%f ", y[i]);
        printf("\n");
    }
}
void Print_matrix(char* title, double local_A[], int local_n, 
                 int n, int my_rank) {
    double* A = NULL;
    if (my_rank == 0) {
        A = malloc(n * n * sizeof(double));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < local_n; j++)
                A[i*n + j] = local_A[i*local_n + j];

        for (int src = 1; src < comm_sz; src++) {
            int start_col = src * local_n;
            for (int i = 0; i < n; i++)
                MPI_Recv(&A[i*n + start_col], local_n, MPI_DOUBLE, 
                        src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        printf("%s\n", title);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                printf("%f ", A[i*n + j]);
            printf("\n");
        }
        free(A);
    } else {
        for (int i = 0; i < n; i++)
            MPI_Send(&local_A[i*local_n], local_n, MPI_DOUBLE, 
                    0, 0, MPI_COMM_WORLD);
    }
}
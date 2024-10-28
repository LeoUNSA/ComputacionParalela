#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
// Function to merge two sorted arrays
void merge(int* arr1, int n1, int* arr2, int n2, int* result) {
    int i = 0, j = 0, k = 0;
    while (i < n1 && j < n2) {
        if (arr1[i] <= arr2[j]) {
            result[k++] = arr1[i++];
        } else {
            result[k++] = arr2[j++];
        }
    }   
    while (i < n1) {
        result[k++] = arr1[i++];
    }
    while (j < n2) {
        result[k++] = arr2[j++];
    }
}
// Function to compare integers for qsort
int compare(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}
int main(int argc, char** argv) {
    int rank, comm_sz;
    int n; // Total number of elements
    int local_n; // Number of elements per process
    int* local_array = NULL;
    int* temp_array = NULL;
    int* all_arrays = NULL;
    int* final_array = NULL;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    // Process 0 reads n and broadcasts it
    if (rank == 0) {
        printf("Enter the total number of elements: ");
        scanf("%d", &n);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Calculate local_n
    local_n = n / comm_sz;
    if (rank < n % comm_sz) {
        local_n++;
    }
    // Allocate memory for arrays
    local_array = (int*)malloc(local_n * sizeof(int));
    temp_array = (int*)malloc(n * sizeof(int));
    if (rank == 0) {
        all_arrays = (int*)malloc(n * sizeof(int));
        final_array = (int*)malloc(n * sizeof(int));
    }
    // Generate random numbers
    srand(time(NULL) + rank); // Different seed for each process
    for (int i = 0; i < local_n; i++) {
        local_array[i] = rand() % 1000; // Random numbers between 0 and 999
    }
    // Sort local array
    qsort(local_array, local_n, sizeof(int), compare);
    // Gather all local arrays to process 0 for initial display
    int* recvcounts = (int*)malloc(comm_sz * sizeof(int));
    int* displs = (int*)malloc(comm_sz * sizeof(int));
    // Calculate receive counts and displacements
    for (int i = 0; i < comm_sz; i++) {
        recvcounts[i] = n / comm_sz;
        if (i < n % comm_sz) {
            recvcounts[i]++;
        }
        displs[i] = (i > 0) ? displs[i-1] + recvcounts[i-1] : 0;
    }
    MPI_Gatherv(local_array, local_n, MPI_INT,
                all_arrays, recvcounts, displs,
                MPI_INT, 0, MPI_COMM_WORLD);
    // Print initial distributed arrays
    if (rank == 0) {
        printf("\nInitial distributed sorted arrays:\n");
        for (int i = 0; i < comm_sz; i++) {
            printf("Process %d: ", i);
            for (int j = 0; j < recvcounts[i]; j++) {
                printf("%d ", all_arrays[displs[i] + j]);
            }
            printf("\n");
        }
    }
    // Tree-structured merge
    int step = 1;
    int local_size = local_n;
    memcpy(temp_array, local_array, local_n * sizeof(int));
    while (step < comm_sz) {
        if (rank % (2 * step) == 0) {
            if (rank + step < comm_sz) {
                // Receive size of incoming array
                MPI_Recv(&local_n, 1, MPI_INT, rank + step,
                        0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // Receive the array
                MPI_Recv(local_array, local_n, MPI_INT, rank + step,
                        0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // Merge the received array with current array
                merge(temp_array, local_size, local_array, local_n, final_array);           
                // Update local size and copy result to temp array
                local_size += local_n;
                memcpy(temp_array, final_array, local_size * sizeof(int));
            }
        } else if (rank % (2 * step) == step) {
            // Send size of local array
            MPI_Send(&local_size, 1, MPI_INT, rank - step,
                    0, MPI_COMM_WORLD);
            
            // Send the array
            MPI_Send(temp_array, local_size, MPI_INT, rank - step,
                    0, MPI_COMM_WORLD);
        }
        step *= 2;
    }
    // Print final sorted array
    if (rank == 0) {
        printf("\nFinal sorted array:\n");
        for (int i = 0; i < n; i++) {
            printf("%d ", temp_array[i]);
        }
        printf("\n");
    }
    // Clean up
    free(local_array);
    free(temp_array);
    free(recvcounts);
    free(displs);
    if (rank == 0) {
        free(all_arrays);
        free(final_array);
    }
    MPI_Finalize();
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
// Function to find the appropriate bin for a data point using binary search
int Find_bin(float data_point, float* bin_maxes, int bin_count, float min_meas) {
    // Special case for first bin
    if (data_point < bin_maxes[0]) return 0;
    
    // Binary search for other bins
    int left = 0, right = bin_count - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (mid == 0 || (data_point >= bin_maxes[mid-1] && data_point < bin_maxes[mid])) {
            return mid;
        }
        if (data_point < bin_maxes[mid]) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return bin_count - 1;
}
int main(int argc, char* argv[]) {
    int rank, size, data_count, bin_count;
    float min_meas, max_meas, *data = NULL, *bin_maxes;
    int *bin_counts = NULL, *local_bin_counts;   
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Process 0 reads input
    if (rank == 0) {
        printf("Enter number of data points: ");
        scanf("%d", &data_count);
        printf("Enter min measurement: ");
        scanf("%f", &min_meas);
        printf("Enter max measurement: ");
        scanf("%f", &max_meas);
        printf("Enter number of bins: ");
        scanf("%d", &bin_count);   
        // Allocate and read data
        data = (float*)malloc(data_count * sizeof(float));
        printf("Enter data points:\n");
        for (int i = 0; i < data_count; i++) {
            scanf("%f", &data[i]);
        }
    }
    // Broadcast parameters to all processes
    MPI_Bcast(&data_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&min_meas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_meas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bin_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Calculate bin boundaries
    float bin_width = (max_meas - min_meas) / bin_count;
    bin_maxes = (float*)malloc(bin_count * sizeof(float));
    for (int b = 0; b < bin_count; b++) {
        bin_maxes[b] = min_meas + bin_width * (b + 1);
    }
    // Calculate local array size and offset
    int local_size = data_count / size;
    int remainder = data_count % size;
    int local_offset = rank * local_size + (rank < remainder ? rank : remainder);
    if (rank < remainder) local_size++;
    // Allocate local arrays
    float* local_data = (float*)malloc(local_size * sizeof(float));
    local_bin_counts = (int*)calloc(bin_count, sizeof(int));
    // Scatter data to all processes
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            int curr_size = data_count / size + (i < remainder ? 1 : 0);
            int offset = i * (data_count / size) + (i < remainder ? i : remainder);
            if (i == 0) {
                // Copy directly for process 0
                memcpy(local_data, data, curr_size * sizeof(float));
            } else {
                MPI_Send(&data[offset], curr_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(local_data, local_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // Process local data
    for (int i = 0; i < local_size; i++) {
        int bin = Find_bin(local_data[i], bin_maxes, bin_count, min_meas);
        local_bin_counts[bin]++;
    }
    // Allocate array for final results on process 0
    if (rank == 0) {
        bin_counts = (int*)calloc(bin_count, sizeof(int));
    }
    // Reduce all local counts to process 0
    MPI_Reduce(local_bin_counts, bin_counts, bin_count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // Print results
    if (rank == 0) {
        printf("\nHistogram results:\n");
        for (int b = 0; b < bin_count; b++) {
            printf("Bin %d (up to %.2f): %d\n", b, bin_maxes[b], bin_counts[b]);
        }
    }
    // Clean up
    free(local_data);
    free(local_bin_counts);
    free(bin_maxes);
    if (rank == 0) {
        free(data);
        free(bin_counts);
    }
    
    MPI_Finalize();
    return 0;
}
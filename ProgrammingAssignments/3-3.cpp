#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
double Tree_sum(double local_sum, int my_rank, int comm_sz, MPI_Comm comm) {
    int phase;
    int partner;
    double partial_sum = local_sum;
    // Tree-structured sum for power of 2 processes
    for (phase = 0; phase < log2(comm_sz); phase++) {
        partner = my_rank ^ (1 << phase);  // XOR to find partner
        
        if (my_rank < partner) {
            MPI_Recv(&local_sum, 1, MPI_DOUBLE, partner, 0, comm, MPI_STATUS_IGNORE);
            partial_sum += local_sum;
        } else {
            MPI_Send(&partial_sum, 1, MPI_DOUBLE, partner, 0, comm);
            break;  // Exit loop after sending
        }
    }   
    return partial_sum;
}
double Tree_sum_any_size(double local_sum, int my_rank, int comm_sz, MPI_Comm comm) {
    int phase;
    int partner;
    double partial_sum = local_sum;
    int active = 1;  // Flag to track if process is still active
    for (phase = 0; phase < ceil(log2(comm_sz)); phase++) {
        if (active) {
            // Calculate potential partner
            partner = my_rank ^ (1 << phase);       
            // Check if partner exists and is valid
            if (partner < comm_sz) {
                if (my_rank < partner) {
                    // Receive and add partner's sum
                    MPI_Recv(&local_sum, 1, MPI_DOUBLE, partner, 0, comm, MPI_STATUS_IGNORE);
                    partial_sum += local_sum;
                } else {
                    // Send partial sum to partner and become inactive
                    MPI_Send(&partial_sum, 1, MPI_DOUBLE, partner, 0, comm);
                    active = 0;
                }
            }
        }
    }
    return partial_sum;
}

int main(int argc, char* argv[]) {
    int comm_sz, my_rank;
    double local_sum, total_sum;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // Generate some local data
    local_sum = my_rank + 1.0;  // Simple example: each process has rank + 1
    // Compute global sum using appropriate method
    if ((comm_sz & (comm_sz - 1)) == 0) {  // Check if comm_sz is power of 2
        total_sum = Tree_sum(local_sum, my_rank, comm_sz, MPI_COMM_WORLD);
    } else {
        total_sum = Tree_sum_any_size(local_sum, my_rank, comm_sz, MPI_COMM_WORLD);
    }
    // Print result from root process
    if (my_rank == 0) {
        printf("Total sum = %f\n", total_sum);
        // Verify result
        double expected_sum = (comm_sz * (comm_sz + 1)) / 2.0;
        printf("Expected sum = %f\n", expected_sum);
    }   
    MPI_Finalize();
    return 0;
}
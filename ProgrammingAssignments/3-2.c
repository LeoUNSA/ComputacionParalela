#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
// Function to generate a random double between -1 and 1
double random_double() {
    return (2.0 * rand() / (double)RAND_MAX) - 1.0;
}
int main(int argc, char* argv[]) {
    int rank, size;
    long long int total_tosses;      // Total number of tosses across all processes
    long long int local_tosses;      // Number of tosses for this process
    long long int local_circle = 0;  // Local hits inside circle
    long long int global_circle;     // Global sum of hits
    double pi_estimate;
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Process 0 reads the total number of tosses
    if (rank == 0) {
        printf("Enter the total number of tosses: ");
        scanf("%lld", &total_tosses);
    }
    // Broadcast total_tosses to all processes
    MPI_Bcast(&total_tosses, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    // Calculate local tosses for this process
    local_tosses = total_tosses / size;
    if (rank < total_tosses % size) {
        local_tosses++; // Distribute remainder among processes
    }
    // Seed random number generator differently for each process
    srand(time(NULL) + rank);
    // Perform Monte Carlo simulation
    for (long long int i = 0; i < local_tosses; i++) {
        double x = random_double();
        double y = random_double();
        double dist_squared = x*x + y*y;
        
        if (dist_squared <= 1.0) {
            local_circle++;
        }
    }
    // Sum up all local_circle values
    MPI_Reduce(&local_circle, &global_circle, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);   
    // Process 0 calculates and prints the result
    if (rank == 0) {
        pi_estimate = 4.0 * global_circle / (double)total_tosses;
        printf("\nEstimated value of pi: %f\n", pi_estimate);
        printf("Actual value of pi: 3.141592653589793...\n");
        printf("Total points: %lld\n", total_tosses);
        printf("Points in circle: %lld\n", global_circle);
    }
    
    MPI_Finalize();
    return 0;
}
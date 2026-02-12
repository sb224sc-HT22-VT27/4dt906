#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <mpi.h>
#include <ctime>
#include <omp.h>

// Compare-exchange operation for parallel odd-even sort
void compare_exchange_low(std::vector<int>& local_numbers, int partner, int n_local) {
    std::vector<int> recv_numbers(n_local);
    
    MPI_Sendrecv(local_numbers.data(), n_local, MPI_INT, partner, 0,
                recv_numbers.data(), n_local, MPI_INT, partner, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // Merge and keep the smaller n_local elements
    std::vector<int> merged;
    merged.reserve(2 * n_local);
    merged.insert(merged.end(), local_numbers.begin(), local_numbers.end());
    merged.insert(merged.end(), recv_numbers.begin(), recv_numbers.end());
    std::sort(merged.begin(), merged.end());
    
    // Keep the lower half
    for (int i = 0; i < n_local; i++) {
        local_numbers[i] = merged[i];
    }
}

void compare_exchange_high(std::vector<int>& local_numbers, int partner, int n_local) {
    std::vector<int> recv_numbers(n_local);
    
    MPI_Sendrecv(local_numbers.data(), n_local, MPI_INT, partner, 0,
                recv_numbers.data(), n_local, MPI_INT, partner, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // Merge and keep the larger n_local elements
    std::vector<int> merged;
    merged.reserve(2 * n_local);
    merged.insert(merged.end(), local_numbers.begin(), local_numbers.end());
    merged.insert(merged.end(), recv_numbers.begin(), recv_numbers.end());
    std::sort(merged.begin(), merged.end());
    
    // Keep the upper half
    for (int i = 0; i < n_local; i++) {
        local_numbers[i] = merged[n_local + i];
    }
}

// Parallel odd-even sort using MPI
void oddeven_sort_parallel(std::vector<int>& local_numbers, int rank, int size)
{
    int n_local = local_numbers.size();
    
    // Perform size iterations (enough to guarantee sorting)
    for (int phase = 0; phase < size; phase++) {
        // Odd phase: processes with odd rank compare with rank+1
        if (phase % 2 == 1) {
            if (rank % 2 == 1 && rank < size - 1) {
                // Odd rank exchanges with right neighbor
                compare_exchange_low(local_numbers, rank + 1, n_local);
            } else if (rank % 2 == 0 && rank > 0) {
                // Even rank exchanges with left neighbor
                compare_exchange_high(local_numbers, rank - 1, n_local);
            }
        }
        // Even phase: processes with even rank compare with rank+1
        else {
            if (rank % 2 == 0 && rank < size - 1) {
                // Even rank exchanges with right neighbor
                compare_exchange_low(local_numbers, rank + 1, n_local);
            } else if (rank % 2 == 1 && rank > 0) {
                // Odd rank exchanges with left neighbor
                compare_exchange_high(local_numbers, rank - 1, n_local);
            }
        }
    }
}

void print_sort_status(std::vector<int> numbers)
{
    std::cout << "The input is sorted?: " << (std::is_sorted(numbers.begin(), numbers.end()) == 0 ? "False" : "True") << std::endl;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    constexpr unsigned int total_size = 100000; // Total number of elements
    int n_local = total_size / size;
    
    std::vector<int> numbers;
    std::vector<int> local_numbers(n_local);
    
    // Process 0 generates and distributes data
    if (rank == 0) {
        numbers.resize(total_size);
        srand(time(0));
        std::generate(numbers.begin(), numbers.end(), rand);
        
        print_sort_status(numbers);
    }
    
    // Distribute data
    MPI_Scatter(rank == 0 ? numbers.data() : nullptr, n_local, MPI_INT,
               local_numbers.data(), n_local, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Sort locally first
    std::sort(local_numbers.begin(), local_numbers.end());
    
    // Start timing
    auto start = std::chrono::steady_clock::now();
    
    // Parallel odd-even sort
    oddeven_sort_parallel(local_numbers, rank, size);
    
    // Gather results
    MPI_Gather(local_numbers.data(), n_local, MPI_INT,
              rank == 0 ? numbers.data() : nullptr, n_local, MPI_INT, 0, MPI_COMM_WORLD);
    
    auto end = std::chrono::steady_clock::now();
    
    // Process 0 prints results
    if (rank == 0) {
        print_sort_status(numbers);
        std::cout << "Elapsed time =  " << std::chrono::duration<double>(end - start).count() << " sec\n";
    }
    
    MPI_Finalize();
    return 0;
}

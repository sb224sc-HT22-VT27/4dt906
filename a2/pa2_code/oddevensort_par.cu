#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>

__global__ void oddeven_sort_singleblock(int *d_data, int n)
{
    int tid    = threadIdx.x;
    int stride = blockDim.x;

    for (int phase = 0; phase < n; phase++) {
        int start = phase & 1;
        for (int j = start + 2 * tid; j < n - 1; j += 2 * stride) {
            if (d_data[j] > d_data[j + 1]) {
                int tmp        = d_data[j];
                d_data[j]      = d_data[j + 1];
                d_data[j + 1]  = tmp;
            }
        }
        __syncthreads();
    }
}

__global__ void oddeven_phase_multiblock(int *d_data, int n, int phase)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int start = phase & 1;
    int j     = start + 2 * idx;
    if (j < n - 1) {
        if (d_data[j] > d_data[j + 1]) {
            int tmp       = d_data[j];
            d_data[j]     = d_data[j + 1];
            d_data[j + 1] = tmp;
        }
    }
}

static bool is_sorted_check(const std::vector<int>& v)
{
    for (size_t i = 0; i + 1 < v.size(); i++)
        if (v[i] > v[i + 1]) return false;
    return true;
}

// Syncs using `__syncthreads()`
// Limited to `1024` threads
// Higher complexity, due to looping
// Better for smaller datasets
static void run_singleblock(const std::vector<int>& h_input)
{
    int n = static_cast<int>(h_input.size());
    std::vector<int> h_data = h_input;

    int *d_data;
    if (cudaMalloc(&d_data, n * sizeof(int)) != cudaSuccess) {
        std::cerr << "cudaMalloc failed in run_singleblock\n"; return;
    }
    cudaMemcpy(d_data, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // CUDA max 1024 threads per block; each thread handles ceil(n/2/1024) pairs
    int threads = std::min(n / 2, 1024);
    if (threads < 1) threads = 1;

    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();

    oddeven_sort_singleblock<<<1, threads>>>(d_data, n);

    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();

    cudaMemcpy(h_data.data(), d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    bool sorted  = is_sorted_check(h_data);
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "[Single-block] n=" << n
              << "  threads=" << threads
              << "  sorted=" << (sorted ? "Yes" : "No")
              << "  Elapsed time = " << elapsed << " sec\n";
}

// Syncs using Kernal Launch Boundary
// Scales to GPU capacity
// Lower complexity, one pair per thread
// Better for larger datasets
static void run_multiblock(const std::vector<int>& h_input)
{
    int n = static_cast<int>(h_input.size());
    std::vector<int> h_data = h_input;

    int *d_data;
    if (cudaMalloc(&d_data, n * sizeof(int)) != cudaSuccess) {
        std::cerr << "cudaMalloc failed in run_multiblock\n"; return;
    }
    cudaMemcpy(d_data, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int threads_per_block = 1024;
    int pairs             = (n + 1) / 2;
    int blocks            = (pairs + threads_per_block - 1) / threads_per_block;

    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();

    for (int phase = 0; phase < n; phase++) {
        oddeven_phase_multiblock<<<blocks, threads_per_block>>>(d_data, n, phase);
    }

    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();

    cudaMemcpy(h_data.data(), d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    bool sorted  = is_sorted_check(h_data);
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "[Multi-block]  n=" << n
              << "  blocks=" << blocks
              << "  sorted=" << (sorted ? "Yes" : "No")
              << "  Elapsed time = " << elapsed << " sec\n";
}

int main(int argc, char **argv)
{
    unsigned int size = 524288; // Default 2^19
    if (argc > 1) {
        try {
            size = static_cast<unsigned int>(std::stoul(argv[1]));
        } catch (...) {
            std::cerr << "Invalid size argument. Using default: " << size << "\n";
        }
    }

    std::vector<int> numbers(size);
    srand(static_cast<unsigned>(time(nullptr)));
    std::generate(numbers.begin(), numbers.end(), rand);

    run_singleblock(numbers);
    run_multiblock(numbers);

    return 0;
}
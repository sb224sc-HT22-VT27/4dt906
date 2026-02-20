#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>

// ─────────────────────────────────────────────────────────────────────────────
// Single-block kernel: all N phases are executed inside ONE kernel launch.
// Threads within the block synchronize after each phase using __syncthreads().
// Each thread may handle more than one pair when N/2 > blockDim.x.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void oddeven_sort_singleblock(int *d_data, int n)
{
    int tid    = threadIdx.x;
    int stride = blockDim.x;

    for (int phase = 0; phase < n; phase++) {
        int start = phase & 1;   // 0 = even phase, 1 = odd phase
        // Each thread covers pairs starting at (start + 2*tid), then strides
        // by 2*stride so all pairs in [start, n-2] are covered collectively.
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

// ─────────────────────────────────────────────────────────────────────────────
// Multi-block kernel: one phase per kernel launch.
// The host launches this kernel N times; implicit global synchronization
// happens between kernel launches.
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// Host-side correctness check
// ─────────────────────────────────────────────────────────────────────────────
static bool is_sorted_check(const std::vector<int>& v)
{
    for (size_t i = 0; i + 1 < v.size(); i++)
        if (v[i] > v[i + 1]) return false;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Run single-block variant and print timing + correctness
// ─────────────────────────────────────────────────────────────────────────────
static void run_singleblock(const std::vector<int>& h_input)
{
    int n = static_cast<int>(h_input.size());
    std::vector<int> h_data = h_input;   // work on a copy

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
              << "  Elapsed time =  " << elapsed << " sec\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Run multi-block variant and print timing + correctness
// ─────────────────────────────────────────────────────────────────────────────
static void run_multiblock(const std::vector<int>& h_input)
{
    int n = static_cast<int>(h_input.size());
    std::vector<int> h_data = h_input;   // work on a copy

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
              << "  Elapsed time =  " << elapsed << " sec\n";
}

int main(int argc, char **argv)
{
    // Default: 2^19 elements as specified by the assignment benchmarking requirement
    int n = (1 << 19);
    if (argc >= 2) n = std::atoi(argv[1]);

    std::vector<int> numbers(n);
    srand(static_cast<unsigned>(time(nullptr)));
    std::generate(numbers.begin(), numbers.end(), rand);

    std::cout << "Odd-Even Sort – CUDA  (n=" << n << ")\n";
    std::cout << "----------------------------------------------\n";

    run_singleblock(numbers);
    run_multiblock(numbers);

    return 0;
}
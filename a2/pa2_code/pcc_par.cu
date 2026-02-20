#define _XOPEN_SOURCE 600
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>

static int ROWS = 128;
static int COLS = 128;

// ─────────────────────────────────────────────────────────────────────────────
// Host: generate matrix with seeded drand48 (identical to pcc_seq)
// ─────────────────────────────────────────────────────────────────────────────
static void generatematrix(double *matrix, unsigned long seed)
{
    srand48((long)seed);
    for (int i = 0; i < ROWS * COLS; i++)
        matrix[i] = drand48();
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: compute row means  (one thread per row)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_means(const double *matrix, double *mean,
                              int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    double sum = 0.0;
    for (int j = 0; j < cols; j++)
        sum += matrix[row * cols + j];
    mean[row] = sum / cols;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: compute mean-adjusted matrix (mm) and std per row
//           (one thread per row)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_mm_std(const double *matrix, const double *mean,
                               double *mm, double *std_dev,
                               int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    double sum = 0.0;
    for (int j = 0; j < cols; j++) {
        double diff          = matrix[row * cols + j] - mean[row];
        mm[row * cols + j]   = diff;
        sum                 += diff * diff;
    }
    std_dev[row] = sqrt(sum);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: compute Pearson correlations.
//
// Grid layout: one block per sample1 (blockIdx.x = sample1).
// Threads within the block stride over all sample2 > sample1.
//
// Output index formula matches pcc_seq exactly:
//   offset = (sample1+1)*(sample1+2)/2
//   output[sample1*rows + sample2 - offset] = r
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_pearson(const double *mm, const double *std_dev,
                                double *output, int rows, int cols)
{
    int sample1 = blockIdx.x;
    if (sample1 >= rows - 1) return;

    // Triangular offset identical to the sequential pcc's inner summ loop
    int tri_offset = (sample1 + 1) * (sample1 + 2) / 2;
    int num_pairs  = rows - sample1 - 1;

    for (int idx = threadIdx.x; idx < num_pairs; idx += blockDim.x) {
        int sample2 = sample1 + 1 + idx;
        double sum  = 0.0;
        for (int k = 0; k < cols; k++)
            sum += mm[sample1 * cols + k] * mm[sample2 * cols + k];
        output[sample1 * rows + sample2 - tri_offset] =
            sum / (std_dev[sample1] * std_dev[sample2]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Write output file (identical format to pcc_seq)
// ─────────────────────────────────────────────────────────────────────────────
static void writeoutput(const double *output, long long cor_size,
                         const char *name)
{
    FILE *f = fopen(name, "wb");
    for (long long i = 0; i < cor_size; i++)
        std::fprintf(f, "%.15g\n", output[i]);
    fclose(f);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s matrix_height matrix_width [seed]\n",
                     argv[0]);
        return -1;
    }

    ROWS = std::atoi(argv[1]);
    if (ROWS < 1) {
        std::fprintf(stderr, "error: height must be at least 1\n");
        return -1;
    }
    COLS = std::atoi(argv[2]);
    if (COLS < 1) {
        std::fprintf(stderr, "error: width must be at least 1\n");
        return -1;
    }

    unsigned long seed = 12345;
    if (argc >= 4) seed = (unsigned long)std::atol(argv[3]);

    char output_filename[50];
    std::snprintf(output_filename, sizeof(output_filename),
                  "pccout_%d_%d.dat", ROWS, COLS);

    long long cor_size = (long long)(ROWS - 1) * ROWS / 2;

    // ── Host allocations ────────────────────────────────────────────────────
    double *h_matrix = (double*)malloc(sizeof(double) * ROWS * COLS);
    double *h_output = (double*)malloc(sizeof(double) * cor_size);
    if (!h_matrix || !h_output) {
        std::fprintf(stderr, "malloc failed\n");
        return 1;
    }

    generatematrix(h_matrix, seed);

    // ── Device allocations ──────────────────────────────────────────────────
    double *d_matrix, *d_mean, *d_mm, *d_std, *d_output;
    if (cudaMalloc(&d_matrix, sizeof(double) * ROWS * COLS) != cudaSuccess ||
        cudaMalloc(&d_mean,   sizeof(double) * ROWS)        != cudaSuccess ||
        cudaMalloc(&d_mm,     sizeof(double) * ROWS * COLS) != cudaSuccess ||
        cudaMalloc(&d_std,    sizeof(double) * ROWS)        != cudaSuccess ||
        cudaMalloc(&d_output, sizeof(double) * cor_size)    != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        free(h_matrix); free(h_output);
        return 1;
    }

    cudaMemcpy(d_matrix, h_matrix, sizeof(double) * ROWS * COLS,
               cudaMemcpyHostToDevice);

    // ── Kernel launches ──────────────────────────────────────────────────────
    int threads    = 256;
    int row_blocks = (ROWS + threads - 1) / threads;

    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();

    kernel_means<<<row_blocks, threads>>>(d_matrix, d_mean, ROWS, COLS);
    kernel_mm_std<<<row_blocks, threads>>>(d_matrix, d_mean,
                                           d_mm, d_std, ROWS, COLS);
    // One block per sample1 row; 256 threads stride over sample2 values
    kernel_pearson<<<ROWS - 1, 256>>>(d_mm, d_std, d_output, ROWS, COLS);

    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();

    // ── Copy result back ─────────────────────────────────────────────────────
    cudaMemcpy(h_output, d_output, sizeof(double) * cor_size,
               cudaMemcpyDeviceToHost);

    std::cout << "Elapsed time =  "
              << std::fixed << std::setprecision(4)
              << std::chrono::duration<double>(t1 - t0).count()
              << " sec\n";

    writeoutput(h_output, cor_size, output_filename);

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_matrix); cudaFree(d_mean);
    cudaFree(d_mm);     cudaFree(d_std);
    cudaFree(d_output);
    free(h_matrix); free(h_output);

    return 0;
}
#define _XOPEN_SOURCE 600
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>

int COLS = 128;
int ROWS = 128;

/**
 * Generate matrix in-place using seeded drand48() for reproducible data.
 **/
void generatematrix(double *matrix, unsigned long seed){
	srand48((long)seed);
	for (int i = 0; i < ROWS * COLS; i++) {
		matrix[i] = drand48();
	}
}

/**
 * Calculate row mean for a range of rows [row_start, row_end).
 */
void calcmean_thread(double *matrix, double *mean, int row_start, int row_end){
	for (int i = row_start; i < row_end; i++){
		double sum = 0.0;
		for (int j = 0; j < COLS; j++){
			sum += matrix[i * COLS + j];
		}
		mean[i] = sum / (double)COLS;
	}
}

/**
 * Calculate matrix - rowmean and standard deviation for a range of rows.
 */
void calc_mm_std_thread(double *matrix, double *mean, double *mm, double *std_dev,
                        int row_start, int row_end){
	for (int i = row_start; i < row_end; i++){
		double sum = 0.0;
		for (int j = 0; j < COLS; j++){
			double diff = matrix[i * COLS + j] - mean[i];
			mm[i * COLS + j] = diff;
			sum += diff * diff;
		}
		std_dev[i] = std::sqrt(sum);
	}
}

/**
 * Calculate Pearson correlations for sample1 rows assigned to this thread.
 * Uses interleaved (strided) row assignment so that heavy rows (many pairs)
 * are spread evenly across threads, avoiding load imbalance.
 * Each thread writes to non-overlapping output indices, so no data races.
 */
void pearson_thread(double *mm, double *std_dev, double *output,
                    int thread_id, int num_threads){
	for (int sample1 = thread_id; sample1 < ROWS - 1; sample1 += num_threads){
		int summ = 0;
		for (int l = 0; l <= sample1 + 1; l++)
			summ += l;

		for (int sample2 = sample1 + 1; sample2 < ROWS; sample2++){
			double sum = 0.0;
			for (int i = 0; i < COLS; i++){
				sum += mm[sample1 * COLS + i] * mm[sample2 * COLS + i];
			}
			output[sample1 * ROWS + sample2 - summ] = sum / (std_dev[sample1] * std_dev[sample2]);
		}
	}
}

void pearson_par(double *input, double *output, int num_threads){

	double *mean    = (double*)malloc(sizeof(double) * ROWS);
	double *std_dev = (double*)malloc(sizeof(double) * ROWS);

	if (mean == NULL || std_dev == NULL){
		std::fprintf(stderr, "Memory allocation failed\n");
		std::exit(1);
	}
	double *minusmean = (double*)malloc(sizeof(double) * ROWS * COLS);
	if (minusmean == NULL) {
		std::fprintf(stderr, "Memory allocation failed\n");
		std::exit(1);
	}

	// Helper: split [0, total) evenly among num_threads (used for linear steps)
	auto make_threads = [&](int total, auto fn) {
		std::vector<std::thread> threads;
		int per = total / num_threads;
		int rem = total % num_threads;
		int start = 0;
		for (int t = 0; t < num_threads; t++){
			int end = start + per + (t < rem ? 1 : 0);
			if (start < end)
				threads.emplace_back(fn, start, end);
			start = end;
		}
		for (auto& th : threads) th.join();
	};

	// Step 1: compute row means in parallel
	make_threads(ROWS, [&](int s, int e){
		calcmean_thread(input, mean, s, e);
	});

	// Step 2: compute (matrix - mean) and std in parallel
	make_threads(ROWS, [&](int s, int e){
		calc_mm_std_thread(input, mean, minusmean, std_dev, s, e);
	});

	// Step 3: compute correlation pairs in parallel.
	// Interleaved assignment ensures balanced work: row 0 (ROWS-1 pairs) and
	// row ROWS-2 (1 pair) go to the same thread, keeping loads equal.
	{
		std::vector<std::thread> threads;
		for (int t = 0; t < num_threads; t++)
			threads.emplace_back(pearson_thread, minusmean, std_dev, output, t, num_threads);
		for (auto& th : threads) th.join();
	}

	free(mean);
	free(minusmean);
	free(std_dev);
}

void writeoutput(double *output, int cor_size, char *name)
{
	FILE *f;

	f = fopen(name,"wb");
	for (int i = 0; i < cor_size; i++) {
		std::fprintf(f, "%.15g\n", output[i]);
	}
	fclose(f);
}

int main(int argc, char **argv){

	if (argc < 3) { std::fprintf(stderr, "usage: %s matrix_height matrix_width [seed]\n", argv[0]); std::exit(-1); }

	ROWS = atoi(argv[1]);
	if (ROWS < 1) { std::fprintf(stderr, "error: height must be at least 1\n"); std::exit(-1); }

	COLS = atoi(argv[2]);
	if (COLS < 1) { std::fprintf(stderr, "error: width must be at least 1\n"); std::exit(-1); }

	unsigned long seed = 12345;
	if (argc >= 4) { seed = (unsigned long)atol(argv[3]); }

	int num_threads = static_cast<int>(std::thread::hardware_concurrency());
	if (num_threads == 0) num_threads = 4;

	//used to generate the correct filename
	char output_filename[50];
	snprintf(output_filename, 50, "pccout_%d_%d.dat", ROWS, COLS);

	//calculates the size of the output
	long long cor_size = ROWS - 1;
	cor_size *= ROWS;
	cor_size /= 2;

	double *matrix, *output;
	output = (double*)malloc(sizeof(double) * cor_size);
	matrix = (double*)malloc(sizeof(double) * COLS * ROWS);

	if (matrix == NULL || output == NULL){
		return(1);
	}

	generatematrix(matrix, seed);

	/* Chrono timer (same style as oddevensort) */
	auto start = std::chrono::steady_clock::now();
	pearson_par(matrix, output, num_threads);
	auto end = std::chrono::steady_clock::now();
	std::cout << "Elapsed time =  " << std::fixed << std::setprecision(4) << std::chrono::duration<double>(end - start).count() << " sec\n";

	writeoutput(output, cor_size, output_filename);

	free(output);
	free(matrix);
	return(0);
}
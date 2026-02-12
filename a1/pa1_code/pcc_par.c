#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

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
 * Calculate row mean
 */
void calcmean(double *matrix, double *mean){
	int i,j;
	double sum;

	#pragma omp parallel for private(j, sum)
	for(i = 0; i < ROWS; i++){
		sum = 0.0;
		for(j = 0; j < COLS; j++){
			sum += matrix[i * COLS + j];
		}
		mean[i] = sum / (double)COLS;
	}
}

/**
 * Calculate matrix - rowmean, and standard deviation for every row 
 */
void calc_mm_std(double *matrix, double *mean, double *mm, double *std){
	int i,j;
	double sum, diff;

	#pragma omp parallel for private(j, sum, diff)
	for(i = 0; i < ROWS; i++){
		sum = 0.0;
		for(j = 0; j < COLS; j++){
			diff = matrix[i * COLS + j] - mean[i];
			mm[i * COLS + j] = diff;
			sum += diff * diff;
		}
		std[i] = sqrt(sum);
	}
}

/**
 * Parallel Pearson correlation calculation
 * Each process computes a subset of the correlation pairs
 */
void pearson_parallel(double *mm, double *std, double *local_output, 
                     int local_start, int local_end, int cor_size){
	int i, sample1, sample2;
	double sum, r;

	// First pass: count how many pairs each sample1 contributes to local work
	int *sample1_count = (int*)calloc(ROWS, sizeof(int));
	int *sample1_offset = (int*)malloc(sizeof(int) * ROWS);
	
	if(sample1_count == NULL || sample1_offset == NULL) {
		fprintf(stderr, "Memory allocation failed in pearson_parallel\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	
	for(sample1 = 0; sample1 < ROWS-1; sample1++){
		int triangle_offset = 0;
		for(int l = 0; l <= sample1+1; l++)
			triangle_offset += l;
		
		for(sample2 = sample1+1; sample2 < ROWS; sample2++){
			int global_idx = sample1 * ROWS + sample2 - triangle_offset;
			if(global_idx >= local_start && global_idx < local_end) {
				sample1_count[sample1]++;
			}
		}
	}
	
	// Compute offsets for each sample1
	sample1_offset[0] = 0;
	for(sample1 = 1; sample1 < ROWS; sample1++){
		sample1_offset[sample1] = sample1_offset[sample1-1] + sample1_count[sample1-1];
	}

	// Now parallelize the outer loop, each thread writes to non-overlapping regions
	#pragma omp parallel for private(sample2, i, sum, r) schedule(dynamic)
	for(sample1 = 0; sample1 < ROWS-1; sample1++){
		int triangle_offset = 0;
		for(int l = 0; l <= sample1+1; l++)
			triangle_offset += l;
		
		int local_idx = sample1_offset[sample1];
		for(sample2 = sample1+1; sample2 < ROWS; sample2++){
			int global_idx = sample1 * ROWS + sample2 - triangle_offset;
			
			// Only compute if this index is in our local range
			if(global_idx >= local_start && global_idx < local_end) {
				sum = 0.0;
				for(i = 0; i < COLS; i++){
					sum += mm[sample1 * COLS + i] * mm[sample2 * COLS + i];
				}
				r = sum / (std[sample1] * std[sample2]);
				local_output[local_idx++] = r;
			}
		}
	}
	
	free(sample1_count);
	free(sample1_offset);
}

void pearson_par(double *input, double *output, int cor_size, int rank, int size){
    
    double *mean = (double*)malloc(sizeof(double) * ROWS);
	double *std  = (double*)malloc(sizeof(double) * ROWS);
	
	if(mean == NULL || std == NULL){
        fprintf(stderr, "Memory allocation failed\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
    double *minusmean = (double*)malloc(sizeof(double) * ROWS * COLS);
	if(minusmean == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
    
    // All processes compute mean and std (small overhead, simplifies code)
    calcmean(input, mean);
	calc_mm_std(input, mean, minusmean, std);
	
	// Divide work among processes
	int local_size = cor_size / size;
	int remainder = cor_size % size;
	int local_start = rank * local_size + (rank < remainder ? rank : remainder);
	int local_count = local_size + (rank < remainder ? 1 : 0);
	int local_end = local_start + local_count;
	
	// Allocate local output
	double *local_output = (double*)malloc(sizeof(double) * local_count);
	if(local_output == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	
	// Compute local portion of correlations
	pearson_parallel(minusmean, std, local_output, local_start, local_end, cor_size);
	
	// Gather results at root
	// First, gather counts and displacements
	int *recvcounts = NULL;
	int *displs = NULL;
	if(rank == 0) {
		recvcounts = (int*)malloc(sizeof(int) * size);
		displs = (int*)malloc(sizeof(int) * size);
	}
	
	MPI_Gather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	if(rank == 0) {
		displs[0] = 0;
		for(int i = 1; i < size; i++) {
			displs[i] = displs[i-1] + recvcounts[i-1];
		}
	}
	
	// Gather the actual correlation values
	MPI_Gatherv(local_output, local_count, MPI_DOUBLE,
	           output, recvcounts, displs, MPI_DOUBLE,
	           0, MPI_COMM_WORLD);

    free(mean);
    free(minusmean);
    free(std);
    free(local_output);
    if(rank == 0) {
    	free(recvcounts);
    	free(displs);
    }
}

void writeoutput(double *output, int cor_size, char *name)
{
	FILE *f;

	f = fopen(name,"wb");
	for (int i = 0; i < cor_size; i++) {
		fprintf(f, "%.15g\n", output[i]);
	}
	fclose(f);
}

int main(int argc, char **argv){
	
	MPI_Init(&argc, &argv);
	
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	if (argc < 3) { 
		if(rank == 0) {
			fprintf(stderr, "usage: %s matrix_height matrix_width [seed]\n", argv[0]); 
		}
		MPI_Finalize();
		exit(-1); 
	}

	ROWS = atoi(argv[1]);
	if (ROWS < 1) { 
		if(rank == 0) {
			fprintf(stderr, "error: height must be at least 1\n"); 
		}
		MPI_Finalize();
		exit(-1); 
	}

	COLS = atoi(argv[2]);
	if (COLS < 1) { 
		if(rank == 0) {
			fprintf(stderr, "error: width must be at least 1\n"); 
		}
		MPI_Finalize();
		exit(-1); 
	}

	unsigned long seed = 12345;
	if (argc >= 4) { seed = (unsigned long)atol(argv[3]); }

	// Buffer size to accommodate filename pattern: "pccout_<rows>_<cols>.dat"
	// Maximum safe buffer for large dimensions (up to 10 digits each)
	#define FILENAME_BUFFER_SIZE 50
	char output_filename[FILENAME_BUFFER_SIZE];
	snprintf(output_filename, FILENAME_BUFFER_SIZE, "pccout_%d_%d.dat", ROWS, COLS);
	
	//calculates the size of the output
	long long cor_size = ROWS - 1;
    cor_size *= ROWS;
    cor_size /= 2;

	double *matrix, *output;
	matrix = (double*)malloc(sizeof(double) * COLS * ROWS);

	if(matrix == NULL){
		MPI_Finalize();
		return(1);
	}
	
	// All processes generate the same matrix (for simplicity)
	generatematrix(matrix, seed);

	// Only root needs output buffer
	if(rank == 0) {
		output = (double*)malloc(sizeof(double) * cor_size);
		if(output == NULL) {
			MPI_Finalize();
			return(1);
		}
	} else {
		output = NULL;
	}

	/* Timing using MPI_Wtime for consistency */
	double start = MPI_Wtime();
	pearson_par(matrix, output, cor_size, rank, size);
	double end = MPI_Wtime();
	
	if(rank == 0) {
		printf("Elapsed time =  %.4f sec\n", end - start);
		writeoutput(output, cor_size, output_filename);	
		free(output);
	}

	free(matrix);
	
	MPI_Finalize();
	return(0);
}

#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

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
 * Compute Pearson correlations for sample1 rows assigned to this process.
 * Uses interleaved (strided) row assignment so that heavy rows (many pairs)
 * are spread evenly across processes, avoiding load imbalance.
 * Process with given rank handles sample1 = rank, rank+size, rank+2*size, ...
 * Each process writes directly to the correct positions in output.
 */
void pearson_parallel(double *mm, double *std, double *output,
                      int rank, int size){
	int i, sample1, sample2;
	double sum;

	for(sample1 = rank; sample1 < ROWS-1; sample1 += size){
		int tri_offset = (sample1 + 1) * (sample1 + 2) / 2;
		for(sample2 = sample1+1; sample2 < ROWS; sample2++){
			sum = 0.0;
			for(i = 0; i < COLS; i++){
				sum += mm[sample1 * COLS + i] * mm[sample2 * COLS + i];
			}
			output[sample1 * ROWS + sample2 - tri_offset] =
				sum / (std[sample1] * std[sample2]);
		}
	}
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
	
	// Each process writes its interleaved rows into a zeroed local buffer
	double *local_output = (double*)calloc(cor_size, sizeof(double));
	if(local_output == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	
	// Compute correlations for this process's interleaved rows
	pearson_parallel(minusmean, std, local_output, rank, size);
	
	// Reduce all local buffers to root; non-computed entries are 0 so sum is correct
	MPI_Reduce(local_output, output, cor_size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    free(mean);
    free(minusmean);
    free(std);
    free(local_output);
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

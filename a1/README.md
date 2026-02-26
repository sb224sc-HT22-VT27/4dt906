# Programming Assignment 1 - Parallel Computing

This directory contains the implementation of Assignment 1 for the 4DT906 Parallel Computing course.

## Contents

- **pa1_code/**: Source code for the assignment
  - `oddevensort.cpp`: Sequential odd-even sort implementation (provided)
  - `oddevensort_par.cpp`: Parallel odd-even sort using C++ `std::thread`
  - `oddevensort_par_mp.cpp`: Parallel odd-even sort using MPI
  - `pcc_seq.cpp`: Sequential Pearson Correlation Coefficient (provided)
  - `pcc_par.cpp`: Parallel Pearson Correlation Coefficient using C++ `std::thread`
  - `pcc_par_mp.c`: Parallel Pearson Correlation Coefficient using MPI
  - `verify.c`: Verification tool for comparing outputs
  - `Makefile`: Build script for all programs
  - `benchmark.sh`: Benchmark script for comparing sequential and parallel performance

- **report.tex**: LaTeX source for the assignment report
- **report.pdf**: Compiled PDF report documenting the implementation and results
- **pa1_26.pdf**: Assignment instructions (provided)

## Building

To build all programs:

```bash
cd pa1_code
make
```

To build specific programs:

```bash
make oddevensort        # Sequential odd-even sort
make oddevensort_par    # Parallel odd-even sort
make pcc_seq            # Sequential PCC
make pcc_par            # Parallel PCC
make verify             # Verification tool
```

## Running

### Odd-Even Sort

Sequential version:

```bash
./oddevensort
```

Parallel version (uses all available hardware threads):

```bash
./oddevensort_par # or mpirun -np 4 ./oddevensort_par_mp (depending on version)
```

### Pearson Correlation Coefficient

Sequential version:

```bash
./pcc_seq <rows> <cols> [seed]
# Example:
./pcc_seq 128 128 42
```

Parallel version (uses all available hardware threads):

```bash
./pcc_par <rows> <cols> [seed] # or mpirun -np 4 ./pcc_par_mp <rows> <cols> [seed] (depending on version)
```

### Benchmark

To run the full benchmark comparing sequential and parallel PCC:

```bash
./benchmark.sh ./pcc_seq ./pcc_par ./verify
```

This will test various matrix sizes (64×64 up to 4096×4096) and report:

- Validation status (correctness)
- Sequential and parallel execution times
- Speedup factor

## Implementation Details

### Odd-Even Transposition Sort

#### C++ `std::thread` Implementation

The parallel implementation uses C++ `std::thread` to distribute compare-swap pairs among threads within each phase. A reusable barrier (compatible with C++17) synchronises all threads between phases, preserving the correctness of the algorithm:

1. Each thread handles every *num_threads*-th pair within the current phase
2. Pairs assigned to different threads are always disjoint, so no data races occur
3. All threads reach the barrier before the next phase begins

#### OpenMP Implementation

The parallel implementation uses MPI to distribute the data among processes. Each process:

1. Receives its portion of data via `MPI_Scatter`
2. Sorts its local data
3. Performs compare-exchange operations with neighboring processes in alternating odd/even phases
4. Gathers results back to root via `MPI_Gather`

### Pearson Correlation Coefficient

#### C++ `std::thread` Implementation

The parallel implementation divides work among threads at each computation stage:

1. **Mean**: each thread computes row means for a contiguous slice of rows
2. **Deviation & std**: each thread computes (row − mean) and the row standard deviation for the same slice
3. **Correlations**: each thread computes all correlation pairs where `sample1` falls in its slice

Because each output index is written by exactly one thread, no synchronisation is needed inside the correlation loop.

#### OpenMP Implementation

The parallel implementation distributes the computation of correlation pairs among processes:

1. All processes compute row means and standard deviations
2. Each process computes a subset of the correlation pairs
3. Results are gathered using `MPI_Gatherv` to handle uneven work distribution

## Requirements

- C++ compiler with C++17 support (g++)
- C compiler with C99 support (gcc)
- MPI implementation (OpenMPI or MPICH)

## Cleaning

To remove all compiled binaries and output files:

```bash
make clean
```

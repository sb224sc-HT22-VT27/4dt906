# Programming Assignment 1 - Parallel Computing

This directory contains the implementation of Assignment 1 for the 4DT906 Parallel Computing course.

## Contents

- **pa1_code/**: Source code for the assignment
  - `oddevensort.cpp`: Sequential odd-even sort implementation (provided)
  - `oddevensort_par.cpp`: Parallel odd-even sort using MPI
  - `pcc_seq.cpp`: Sequential Pearson Correlation Coefficient (provided)
  - `pcc_par.c`: Parallel Pearson Correlation Coefficient using MPI
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

Parallel version (example with 4 processes):
```bash
mpirun -np 4 ./oddevensort_par
```

### Pearson Correlation Coefficient

Sequential version:
```bash
./pcc_seq <rows> <cols> [seed]
# Example:
./pcc_seq 128 128 42
```

Parallel version:
```bash
mpirun -np 4 ./pcc_par <rows> <cols> [seed]
# Example:
mpirun -np 4 ./pcc_par 128 128 42
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

The parallel implementation uses MPI to distribute the data among processes. Each process:
1. Receives its portion of data via `MPI_Scatter`
2. Sorts its local data
3. Performs compare-exchange operations with neighboring processes in alternating odd/even phases
4. Gathers results back to root via `MPI_Gather`

### Pearson Correlation Coefficient

The parallel implementation distributes the computation of correlation pairs among processes:
1. All processes compute row means and standard deviations
2. Each process computes a subset of the correlation pairs
3. Results are gathered using `MPI_Gatherv` to handle uneven work distribution

## Performance Results

With 4 processes, the parallel PCC implementation achieves:
- **Speedup**: 2.00× to 3.56× depending on problem size
- **Efficiency**: 50% to 89%, improving with larger matrices
- **Best speedup**: 3.56× for 4096×4096 matrices

See `report.pdf` for detailed analysis and discussion.

## Requirements

- C++ compiler with C++17 support (g++)
- C compiler with C99 support (gcc)
- MPI implementation (OpenMPI or MPICH)
- LaTeX (for rebuilding the report)

## Cleaning

To remove all compiled binaries and output files:

```bash
make clean
```

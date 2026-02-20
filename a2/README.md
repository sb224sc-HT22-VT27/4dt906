# Programming Assignment 2 - GPU Parallelization with CUDA

This directory contains the CUDA implementation of Assignment 2 for the
4DT906 Parallel Computing course.

## Contents

- **pa2_code/**: Source code for the assignment
  - `oddevensort_par.cu`: CUDA odd-even sort (single-block and multi-block variants)
  - `pcc_par.cu`: CUDA Pearson Correlation Coefficient
  - `Makefile`: Build script for all programs
  - `benchmark.sh`: Benchmark script comparing sequential and CUDA performance
  - `pa1_code/`: Sequential reference implementations (from Assignment 1)
    - `oddevensort.cpp`: Sequential odd-even sort
    - `pcc_seq.cpp`: Sequential Pearson Correlation Coefficient
    - `verify.c`: Floating-point output verifier

- **report.tex**: LaTeX source for the assignment report
- **pa2_26.pdf**: Assignment instructions

## Requirements

- CUDA Toolkit 12.x or later (`nvcc`)
- NVIDIA GPU (Compute Capability ≥ 7.5 recommended)
- C++ compiler with C++17 support (g++)
- C compiler with C99 support (gcc)
- LaTeX (for rebuilding the report)

## Building

```bash
cd pa2_code
make
```

Individual targets:
```bash
make oddevensort_seq   # Sequential odd-even sort (reference)
make pcc_seq           # Sequential PCC (reference)
make oddevensort_par   # CUDA odd-even sort (both variants)
make pcc_par           # CUDA PCC
make verify            # Verification tool
```

## Running

### Odd-Even Sort

Sequential reference:
```bash
./oddevensort_seq
```

CUDA version (runs both single-block and multi-block variants):
```bash
./oddevensort_par              # default: n=2^19
./oddevensort_par 65536        # custom size
```

### Pearson Correlation Coefficient

Sequential reference:
```bash
./pcc_seq <rows> <cols> [seed]
# Example:
./pcc_seq 128 128 42
```

CUDA version:
```bash
./pcc_par <rows> <cols> [seed]
# Example:
./pcc_par 128 128 42
```

### Benchmarks

```bash
# Odd-even sort (single-block vs multi-block)
make benchmark-odd

# PCC (sequential vs CUDA) with correctness validation
make benchmark-pcc
```

Or run the benchmark script directly:
```bash
./benchmark.sh odd ./oddevensort_seq ./oddevensort_par
./benchmark.sh pcc ./pcc_seq ./pcc_par ./verify
```

## CUDA Implementation Details

### Odd-Even Sort – Single-Block Variant

- Launches exactly **one CUDA block** with up to 1024 threads.
- All N phases execute inside a single kernel launch.
- Threads synchronize after each phase using `__syncthreads()`.
- Each thread handles multiple pairs (via stride loop) when N/2 > 1024.

### Odd-Even Sort – Multi-Block Variant

- Launches **one kernel per phase** (N kernel launches total).
- Each kernel uses ⌈N/2/1024⌉ blocks, giving one thread per comparison pair.
- Global synchronization is implicit between consecutive kernel launches.

### Pearson Correlation Coefficient

Three sequential kernel launches:
1. **`kernel_means`**: one thread per row, computes row mean.
2. **`kernel_mm_std`**: one thread per row, computes mean-adjusted values and std.
3. **`kernel_pearson`**: one block per `sample1` row; threads stride over `sample2` values and compute the inner product for each pair.

Output format is identical to `pcc_seq` and validated with the provided `verify` tool.

## Performance Summary

| Algorithm | Sequential | CUDA | Speedup |
|-----------|-----------|------|---------|
| PCC 4096×4096 | ~129 s | ~9.3 s | ~13.8× |
| Odd-even (single-block, n=2^19) | ~N/A | ~10.2 s | — |
| Odd-even (multi-block, n=2^19) | ~N/A | ~210 s | — |

The single-block sort is ~20× faster than multi-block for large arrays due to
kernel-launch overhead.  See `report.tex` (or the compiled PDF) for full analysis.

# CUDA parallel versions; sequential references live in pa1_code/.

CC       = gcc
CXX      = g++
NVCC     = nvcc
NVCCFLAGS = -O2 -std=c++14 -arch=sm_86

PA1 = pa1_code

all: oddevensort pcc_seq oddevensort_par pcc_par verify

# --- Sequential (given) ---
oddevensort: $(PA1)/oddevensort.cpp
	$(CXX) $< -std=c++17 -O2 -o $@

pcc_seq: $(PA1)/pcc_seq.cpp
	$(CXX) $< -lm -std=c++17 -O2 -o $@

# --- CUDA parallel programs ---
oddevensort_par: oddevensort_par.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

pcc_par: pcc_par.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# --- PCC output verifier (floating-point comparison) ---
verify: $(PA1)/verify.c
	$(CC) $< -lm -std=c99 -o $@ -DQUIET

# --- Benchmarks ---
benchmark-pcc: pcc_seq pcc_par verify
	./benchmark.sh ./pcc_seq ./pcc_par ./verify

clean:
	rm -rf oddevensort oddevensort_par pcc_seq pcc_par verify *.dat
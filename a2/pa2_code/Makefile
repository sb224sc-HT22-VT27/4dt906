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

benchmark-odd: oddevensort oddevensort_par
	@for N in 100000 524288; do \
	  echo -e "\n=== Size: $$N ==="; \
	  echo "--- Sequential Output ---"; \
	  T1=$$(./oddevensort $$N 2>&1 | tee /dev/tty | grep "Elapsed time" | awk '{print $$(NF-1)}'); \
	  echo "--- CUDA Parallel Output ---"; \
	  T2=$$(./oddevensort_par $$N 2>&1 | tee /dev/tty | grep "Multi-block" | awk '{print $$(NF-1)}'); \
	  echo "Speedup (size=$$N): $$(echo "scale=2; $$T1 / $$T2" | bc)x"; \
	done

clean:
	rm -rf oddevensort oddevensort_par pcc_seq pcc_par verify *.dat
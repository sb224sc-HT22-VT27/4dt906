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

# --- Parallell (added) ---
oddevensort_par: oddevensort_par.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

pcc_par: pcc_par.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# PCC output verifier (floating-point comparison)
verify: $(PA1)/verify.c
	$(CC) $< -lm -std=c99 -o $@ -DQUIET

# Run benchmark: validate correctness and report speedup (seq vs par)
benchmark-pcc: pcc_seq pcc_par verify
	./benchmark.sh ./pcc_seq ./pcc_par ./verify

benchmark-odd: oddevensort oddevensort_par
	@for N in 100000 524288; do \
	  echo -e "\n=== Size: $$N ==="; \
	  echo "--- Sequential Output ---"; \
	  T_SEQ=$$(./oddevensort $$N 2>&1 | tee /dev/tty | grep "Elapsed time" | awk '{print $$(NF-1)}'); \
	  echo -e "\n--- CUDA Parallel Output ---"; \
	  PAR_OUT=$$(./oddevensort_par $$N 2>&1 | tee /dev/tty); \
	  T_SB=$$(echo "$$PAR_OUT" | grep "Single-block" | awk '{print $$(NF-1)}'); \
	  T_MB=$$(echo "$$PAR_OUT" | grep "Multi-block"  | awk '{print $$(NF-1)}'); \
	  echo "Speedup single-block (size=$$N): $$(echo "scale=2; $$T_SEQ / $$T_SB" | bc)x"; \
	  echo "Speedup multi-block  (size=$$N): $$(echo "scale=2; $$T_SEQ / $$T_MB" | bc)x"; \
	done

clean:
	rm -rf oddevensort oddevensort_par pcc_seq pcc_par verify *.dat
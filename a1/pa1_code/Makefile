# Sequential versions are provided. Students parallelize and add *_par sources.

HAS_MPI := $(shell command -v mpic++ >/dev/null 2>&1 && echo yes)

CC			= gcc
CXX			= g++
CFLAGS 		= -O2
CXXFLAGS 	= -O2 -std=c++17

# Default list
TARGETS = oddevensort pcc_seq oddevensort_par pcc_par verify

# Add MPI targets only if detected
ifeq ($(HAS_MPI),yes)
    $(info MPI detected.)
    CCP         = mpicc
    CXXP        = mpic++
    TARGETS    += oddevensort_par_mp pcc_par_mp
else
    $(info MPI not found.)
endif

all: $(TARGETS)

# --- Sequential (given) ---
oddevensort: oddevensort.cpp
	$(CXX) $(CXXFLAGS) oddevensort.cpp -o oddevensort

pcc_seq: pcc_seq.cpp
	$(CXX) $(CXXFLAGS) pcc_seq.cpp -lm -o pcc_seq

# --- Parallell (added) ---
oddevensort_par: oddevensort_par.cpp
	$(CXX) $(CXXFLAGS) oddevensort_par.cpp -pthread -o oddevensort_par

pcc_par: pcc_par.cpp
	$(CXX) $(CXXFLAGS) pcc_par.cpp -lm -pthread -o pcc_par

# PCC output verifier (floating-point comparison)
verify: verify.c
	$(CC) $(CFLAGS) verify.c -lm -std=c99 -o verify -DQUIET

# Run benchmark: validate correctness and report speedup (seq vs par)
benchmark-pcc: pcc_seq pcc_par verify
	./benchmark.sh ./pcc_seq ./pcc_par ./verify

oddeven-test: oddevensort oddevensort_par
	@echo "Running benchmarks..."
	@echo "--- Sequential Output ---"; \
	 T1=$$(./oddevensort | tee /dev/tty | grep "Elapsed time" | awk '{print $$4}'); \
	 echo "\n--- Parallel Output ---"; \
	 T2=$$(./oddevensort_par | tee /dev/tty | grep "Elapsed time" | awk '{print $$4}'); \
	 echo "\nSpeedup: $$(echo "scale=2; $$T1 / $$T2" | bc)x"


# OpenMPI for *_mp files
ifeq ($(HAS_MPI),yes)
oddevensort_par_mp: oddevensort_par_mp.cpp
	$(CXXP) $(CXXFLAGS) oddevensort_par_mp.cpp -o oddevensort_par_mp

pcc_par_mp:  pcc_par_mp.c
	$(CCP) $(CFLAGS) pcc_par_mp.c -lm -std=c99 -o pcc_par_mp

benchmark-pcc-mp: pcc_seq pcc_par verify
	./benchmark.sh ./pcc_seq ./pcc_par_mp ./verify

oddeven-test-mp: oddevensort oddevensort_par_mp
	@echo "Running benchmarks..."
	@echo "--- Sequential Output ---"; \
	 T1_RAW=$$(./oddevensort | tee /dev/tty | grep "Elapsed time" | awk '{print $$4}'); \
	 T1=$$(printf "%.6f" $$T1_RAW); \
	 echo "\n--- Parallel Output ---"; \
	 T2_RAW=$$(./oddevensort_par_mp | tee /dev/tty | grep "Elapsed time" | awk '{print $$4}'); \
	 T2=$$(printf "%.6f" $$T2_RAW); \
	 echo "\nSpeedup: $$(echo "scale=2; $$T1 / $$T2" | bc)x"
endif

clean:
	rm -rf $(TARGETS) *.dat

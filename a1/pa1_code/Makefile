# Sequential versions are provided. Students parallelize and add *_par sources.

CC			= gcc
CXX			= g++
CCP 		= mpicc
CXXP 		= mpic++
CFLAGS 		= -O2
CXXFLAGS 	= -O2 -std=c++17

all: oddevensort pcc_seq oddevensort_par pcc_par verify

# --- Sequential (given) ---
oddevensort: oddevensort.cpp
	$(CXX) $(CXXFLAGS) oddevensort.cpp -o oddevensort

pcc_seq: pcc_seq.cpp
	$(CXX) $(CXXFLAGS) pcc_seq.cpp -lm -o pcc_seq

# --- Parallell (added) ---
oddevensort_par: oddevensort_par.cpp
	$(CXX) $(CXXFLAGS) oddevensort_par.cpp -pthread -o oddevensort_par

oddevensort_par_mp: oddevensort_par_mp.cpp
	$(CXXP) $(CXXFLAGS) oddevensort_par_mp.cpp -o oddevensort_par_mp

pcc_par: pcc_par.cpp
	$(CXX) $(CXXFLAGS) pcc_par.cpp -lm -pthread -o pcc_par

pcc_par_mp:  pcc_par_mp.c
	$(CCP) $(CFLAGS) pcc_par_mp.c -lm -std=c99 -o pcc_par_mp

# PCC output verifier (floating-point comparison)
verify: verify.c
	$(CC) $(CFLAGS) verify.c -lm -std=c99 -o verify -DQUIET

# Run benchmark: validate correctness and report speedup (seq vs par)
benchmark-pcc: pcc_seq pcc_par verify
	./benchmark.sh ./pcc_seq ./pcc_par ./verify

clean:
	rm -rf oddevensort oddevensort_par oddevensort_par_mp pcc_seq pcc_par pcc_par_mp verify *.dat

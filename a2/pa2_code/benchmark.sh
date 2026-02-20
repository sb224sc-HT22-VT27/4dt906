#!/bin/bash
# benchmark.sh – run correctness + performance benchmarks for A2 (CUDA).
#
# Usage:
#   ./benchmark.sh odd  <seq_exe> <par_exe>
#   ./benchmark.sh pcc  <seq_exe> <par_exe> [verify_exe]

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
END="\033[0m"

SEED=42

if [ "$#" -lt 3 ]; then
    echo "Usage:"
    echo "  $0 odd <sequential_exe> <cuda_exe>"
    echo "  $0 pcc <sequential_exe> <cuda_exe> [verify_exe]"
    exit 1
fi

MODE=$1
seq_exe=$2
par_exe=$3
verify_exe=${4:-./verify}

# ─────────────────────────────────────────────────────────────────────────────
# Odd-Even Sort benchmark
# ─────────────────────────────────────────────────────────────────────────────
if [ "$MODE" = "odd" ]; then
    echo "Odd-Even Sort benchmark: sequential vs CUDA (single-block & multi-block)"
    echo "-------------------------------------------------------------------------"

    for n in 1024 4096 16384 65536 131072 262144 524288; do
        echo ""
        echo "n=$n"

        # Sequential: parse "Elapsed time =  X sec" from sequential output
        # The sequential executable uses hardcoded size 100000; skip seq for
        # sizes other than 100000 and just report CUDA timings.
        seq_time="N/A"
        if [ "$n" -eq 100000 ]; then
            seq_output=$($seq_exe 2>/dev/null)
            seq_time=$(echo "$seq_output" | grep "Elapsed time" \
                       | sed -n 's/.*Elapsed time =  *\([0-9.]*\).*/\1/p')
        fi

        # CUDA (runs both single-block and multi-block and prints both)
        par_output=$($par_exe $n 2>/dev/null)

        sb_time=$(echo "$par_output" | grep "Single-block" \
                  | sed -n 's/.*Elapsed time =  *\([0-9.]*\).*/\1/p')
        mb_time=$(echo "$par_output" | grep "Multi-block" \
                  | sed -n 's/.*Elapsed time =  *\([0-9.]*\).*/\1/p')

        sb_sorted=$(echo "$par_output" | grep "Single-block" \
                    | sed -n 's/.*sorted=\([A-Za-z]*\).*/\1/p')
        mb_sorted=$(echo "$par_output" | grep "Multi-block" \
                    | sed -n 's/.*sorted=\([A-Za-z]*\).*/\1/p')

        # Correctness
        if [ "$sb_sorted" = "Yes" ]; then
            echo -e "  Single-block: ${GREEN}Correct${END}  time=${sb_time}s"
        else
            echo -e "  Single-block: ${RED}WRONG${END}  time=${sb_time}s"
        fi
        if [ "$mb_sorted" = "Yes" ]; then
            echo -e "  Multi-block:  ${GREEN}Correct${END}  time=${mb_time}s"
        else
            echo -e "  Multi-block:  ${RED}WRONG${END}  time=${mb_time}s"
        fi

        # Speedup single vs multi
        if [ -n "$sb_time" ] && [ -n "$mb_time" ]; then
            speedup=$(awk -v s="$sb_time" -v m="$mb_time" \
                      'BEGIN{if(m>0 && s!="") printf "%.2f",m/s; else print "N/A"}')
            echo -e "  ${YELLOW}Single-block speedup over multi-block: ${speedup}x${END}"
        fi
    done

    echo ""
    echo "Done."
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# PCC benchmark
# ─────────────────────────────────────────────────────────────────────────────
if [ "$MODE" = "pcc" ]; then
    echo "PCC benchmark: sequential vs CUDA"
    echo "----------------------------------"

    for n in 64 128 256 512 1024 2048 4096; do
        echo ""
        echo "n=$n (matrix ${n}x${n})"

        # Sequential run
        seq_output=$($seq_exe $n $n $SEED 2>/dev/null)
        seq_time=$(echo "$seq_output" | grep "Elapsed time" \
                   | sed -n 's/.*Elapsed time =  *\([0-9.]*\).*/\1/p')

        if [ ! -f "pccout_${n}_${n}.dat" ]; then
            echo -e "  Validation: ${RED}Failed (seq produced no output)${END}"
            continue
        fi
        mv "pccout_${n}_${n}.dat" pccout_seq.dat

        # CUDA run
        par_output=$($par_exe $n $n $SEED 2>/dev/null)
        par_time=$(echo "$par_output" | grep "Elapsed time" \
                   | sed -n 's/.*Elapsed time =  *\([0-9.]*\).*/\1/p')

        if [ ! -f "pccout_${n}_${n}.dat" ]; then
            echo -e "  Validation: ${RED}Failed (CUDA produced no output)${END}"
        else
            mv "pccout_${n}_${n}.dat" pccout_par.dat
            if [ -x "$verify_exe" ]; then
                $verify_exe pccout_seq.dat pccout_par.dat 2>/dev/null
                verify_ret=$?
                if [ "$verify_ret" -eq 0 ]; then
                    echo -e "  Validation: ${GREEN}Passed${END}"
                elif [ "$verify_ret" -eq 1 ]; then
                    echo -e "  Validation: ${YELLOW}Passed (within tolerance)${END}"
                else
                    echo -e "  Validation: ${RED}Failed (output differs)${END}"
                fi
            else
                if diff -q pccout_seq.dat pccout_par.dat > /dev/null 2>&1; then
                    echo -e "  Validation: ${GREEN}Passed${END}"
                else
                    echo -e "  Validation: ${RED}Failed (output differs)${END}"
                fi
            fi
        fi

        speedup=$(awk -v s="$seq_time" -v p="$par_time" \
                  'BEGIN{if(p>0 && s!="") printf "%.2f",s/p; else print "N/A"}')
        printf "  Seq time: %ss\n  CUDA time: %ss\n  ${YELLOW}Speedup: %sx${END}\n" \
               "$seq_time" "$par_time" "$speedup"
    done

    echo ""
    echo "Done."
    exit 0
fi

echo "Unknown mode: $MODE (use 'odd' or 'pcc')"
exit 1
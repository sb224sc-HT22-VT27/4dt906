#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>

// CPU pause hint: reduces spin-wait power/contention on x86; falls back to
// yield() on other architectures so threads don't starve the OS scheduler.
#if defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#  define cpu_pause() _mm_pause()
#elif defined(__aarch64__)
#  define cpu_pause() asm volatile("yield" ::: "memory")
#else
#  define cpu_pause() std::this_thread::yield()
#endif

// Fast reusable barrier using atomic spin-wait.
// arrive_and_wait() returns true if any thread called mark_swap() since the
// previous barrier crossing, enabling early exit when the array is sorted.
class Barrier {
    int total;
    std::atomic<int> count;
    std::atomic<int> generation;
    std::atomic<bool> any_swap_this_phase;
    // Written by the last arriving thread under release, read by others under
    // acquire (via the generation counter), so no atomic needed here.
    bool phase_had_swap;
public:
    explicit Barrier(int n)
        : total(n), count(n), generation(0),
          any_swap_this_phase(false), phase_had_swap(true) {}

    // Call before arrive_and_wait() if this thread performed a swap.
    void mark_swap() {
        any_swap_this_phase.store(true, std::memory_order_relaxed);
    }

    // Synchronise all threads.  Returns true if any thread called mark_swap()
    // during the phase that just completed; false means the array is sorted.
    bool arrive_and_wait() {
        int gen = generation.load(std::memory_order_acquire);
        if (count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // Last thread: capture result, reset flag, then release others.
            phase_had_swap = any_swap_this_phase.load(std::memory_order_relaxed);
            any_swap_this_phase.store(false, std::memory_order_relaxed);
            count.store(total, std::memory_order_release);
            generation.fetch_add(1, std::memory_order_release);
        } else {
            while (generation.load(std::memory_order_acquire) == gen)
                cpu_pause();
        }
        // phase_had_swap is safely visible to all threads: the last thread
        // wrote it before the release on generation, and every other thread
        // reads generation with acquire before reading phase_had_swap.
        return phase_had_swap;
    }
};

// Each thread handles a contiguous block of compare-swap pairs per phase.
// Contiguous access avoids false sharing between threads on the same cache lines.
void oddeven_sort_thread(std::vector<int>& numbers, int thread_id, int num_threads,
                          int n, Barrier& barrier)
{
    for (int phase = 1; phase <= n; phase++) {
        int start = phase % 2;
        // Number of compare-swap pairs active in this phase.
        // Valid j: start, start+2, ..., last j < n-1  →  count = (n - start) / 2
        int num_pairs = (n - start) / 2;
        // Divide pairs into contiguous blocks, one per thread
        int per = num_pairs / num_threads;
        int rem = num_pairs % num_threads;
        int pair_start = thread_id * per + (thread_id < rem ? thread_id : rem);
        int pair_end   = pair_start + per + (thread_id < rem ? 1 : 0);
        bool local_swapped = false;
        for (int k = pair_start; k < pair_end; k++) {
            int j = start + 2 * k;
            if (numbers[j] > numbers[j + 1]) {
                std::swap(numbers[j], numbers[j + 1]);
                local_swapped = true;
            }
        }
        if (local_swapped)
            barrier.mark_swap();
        // Early exit: if no thread swapped anything, the array is sorted.
        if (!barrier.arrive_and_wait())
            break;
    }
}

void print_sort_status(std::vector<int> numbers)
{
    std::cout << "The input is sorted?: " << (std::is_sorted(numbers.begin(), numbers.end()) == 0 ? "False" : "True") << std::endl;
}

int main()
{
    constexpr unsigned int size = 1 << 19; // Number of elements in the input
    int num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads == 0) num_threads = 4;

    std::vector<int> numbers(size);
    srand(time(0));
    std::generate(numbers.begin(), numbers.end(), rand);

    print_sort_status(numbers);

    Barrier barrier(num_threads);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    auto start = std::chrono::steady_clock::now();

    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back(oddeven_sort_thread, std::ref(numbers), t, num_threads,
                             static_cast<int>(size), std::ref(barrier));
    }
    for (auto& th : threads) {
        th.join();
    }

    auto end = std::chrono::steady_clock::now();

    print_sort_status(numbers);
    std::cout << "Elapsed time =  " << std::chrono::duration<double>(end - start).count() << " sec\n";

    return 0;
}
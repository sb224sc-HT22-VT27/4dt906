#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>

// Fast reusable barrier using atomic spin-wait (avoids OS scheduler overhead
// of condition_variable for the many short phases in odd-even sort).
class Barrier {
    int total;
    std::atomic<int> count;
    std::atomic<int> generation;
public:
    explicit Barrier(int n) : total(n), count(n), generation(0) {}

    void arrive_and_wait() {
        int gen = generation.load(std::memory_order_acquire);
        if (count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            count.store(total, std::memory_order_release);
            generation.fetch_add(1, std::memory_order_release);
        } else {
            while (generation.load(std::memory_order_acquire) == gen)
                std::this_thread::yield();
        }
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
        for (int k = pair_start; k < pair_end; k++) {
            int j = start + 2 * k;
            if (numbers[j] > numbers[j + 1]) {
                std::swap(numbers[j], numbers[j + 1]);
            }
        }
        barrier.arrive_and_wait();
    }
}

void print_sort_status(std::vector<int> numbers)
{
    std::cout << "The input is sorted?: " << (std::is_sorted(numbers.begin(), numbers.end()) == 0 ? "False" : "True") << std::endl;
}

int main(int argc, char** argv)
{
    unsigned int size = 1u << 19;
    if (argc > 1) {
        try {
            size = static_cast<unsigned int>(std::stoul(argv[1]));
        } catch (...) {
            std::cerr << "Invalid size argument. Using default: " << size << "\n";
        }
    }
    int num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads == 0) num_threads = 4;

    std::vector<int> numbers(size);
    srand(time(0));
    std::generate(numbers.begin(), numbers.end(), rand);

    std::cout << "Size: " << size << "\n";
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
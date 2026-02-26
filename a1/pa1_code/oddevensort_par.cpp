#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

// Reusable barrier for C++17 (std::barrier is C++20)
class Barrier {
    unsigned int count;
    unsigned int waiting;
    unsigned int generation;
    std::mutex mtx;
    std::condition_variable cv;
public:
    explicit Barrier(unsigned int n) : count(n), waiting(0), generation(0) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mtx);
        unsigned int gen = generation;
        if (++waiting == count) {
            generation++;
            waiting = 0;
            cv.notify_all();
        } else {
            cv.wait(lock, [this, gen] { return gen != generation; });
        }
    }
};

// Each thread handles every num_threads-th compare-swap pair in each phase.
// Pairs processed by different threads are disjoint, so no data races occur.
void oddeven_sort_thread(std::vector<int>& numbers, int thread_id, int num_threads,
                          int n, Barrier& barrier)
{
    for (int phase = 1; phase <= n; phase++) {
        int start = phase % 2;
        for (int j = start + thread_id * 2; j < n - 1; j += num_threads * 2) {
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

int main()
{
    constexpr unsigned int size = 100000; // Number of elements in the input
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

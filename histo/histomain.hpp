#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <cstring>

static constexpr std::size_t NUM_VALUES = 256;
using value_t = unsigned char;

#ifdef CUDA
#define CUCH(status)  do { cudaError_t err = status; if (err != cudaSuccess) std::cerr << __FILE__ ":" << __LINE__ << ": error: " << cudaGetErrorString(err) << "\n\t" #status << std::endl, exit(err); } while (false)

static constexpr std::size_t BLOCK_SIZE = 64;
static constexpr std::size_t ELEMS_PER_THREAD = 1024;
static constexpr std::size_t ELEMS_PER_BLOCK = BLOCK_SIZE * ELEMS_PER_THREAD;

static constexpr std::size_t NUM_COPIES = 4;

namespace {
	// Only "unsigned long long" is supported by `atomicAdd`.
	template<typename T>
	__device__ auto atomicAdd(T *ptr, unsigned long long int val)
		-> std::enable_if_t<(sizeof(T) == sizeof(unsigned long long int)), decltype(atomicAdd((unsigned long long int *)ptr, val))> {
		::atomicAdd((unsigned long long int *)ptr, val);
	}
}

#endif

extern void histo(void *in_ptr, std::size_t size, void *out_ptr);

using namespace std::literals::chrono_literals;

int main(int argc, char **argv) {
	if(argc != 3) {
		std::cerr << "Usage: histo <filename> <size>" << std::endl;
		std::abort();
	}

	std::size_t size = std::stoll(argv[2]);

	void *data;

#ifdef CUDA
	CUCH(cudaMallocManaged(&data, size));
#else
	data = std::malloc(size);
#endif

	std::FILE *file = std::fopen(argv[1], "r");
	if(std::fread(data, 1, size, file) != size) {
		std::cerr << "Input error" << std::endl;
		std::abort();
	}
	std::fclose(file);

	std::size_t *counts;

#ifdef CUDA
	CUCH(cudaMallocManaged(&counts, NUM_VALUES * sizeof(std::size_t)));
#else
	counts = (std::size_t *)std::malloc(NUM_VALUES * sizeof(std::size_t));
#endif

	std::memset(counts, 0, NUM_VALUES * sizeof(std::size_t));

	auto t0 = std::chrono::steady_clock::now();
	histo(data, size, counts);
	auto t1 = std::chrono::steady_clock::now();
	std::fprintf(stderr, "%lu.%03u ms\n", (unsigned long) ((t1 - t0) / 1ms), (unsigned) ((t1 - t0) / 1us % 1000));

#ifdef CUDA
	CUCH(cudaFree(data));
#else
	std::free(data);
#endif

	for(std::size_t v = 0; v < NUM_VALUES; v++) {
		std::printf("%12zu * 0x%02x", counts[v], (unsigned) v);
		if(v >= ' ' && v <= '~')
			std::printf(" ('%c')", (char) v);
		std::putchar('\n');
	}

#ifdef CUDA
	CUCH(cudaFree(counts));
#else
	std::free(counts);
#endif
}

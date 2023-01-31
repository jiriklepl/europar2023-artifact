#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <cstring>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/cuda_traverser.cuh>
#include <noarr/structures/interop/cuda_striped.cuh>
#include <noarr/structures/interop/cuda_step.cuh>

#define CUCH(status)  do { cudaError_t err = status; if (err != cudaSuccess) std::cerr << __FILE__ ":" << __LINE__ << ": error: " << cudaGetErrorString(err) << "\n\t" #status << std::endl, exit(err); } while (false)

static constexpr std::size_t NUM_VALUES = 256;

static constexpr std::size_t BLOCK_SIZE = 64;
static constexpr std::size_t ELEMS_PER_THREAD = 1024;
static constexpr std::size_t ELEMS_PER_BLOCK = BLOCK_SIZE * ELEMS_PER_THREAD;

static constexpr std::size_t NUM_COPIES = 4;

using value_t = unsigned char;

// TODO fix this somehow
namespace {
	// For some reason, "unsigned long long" is different from "unsigned long", although both have the same size.
	// Only "unsigned long long" is supported by `atomicAdd`.
	// Unfortunately, "size_t" is defined as "unsigned long", and is thus not supported.
	__device__ std::size_t atomicAdd(std::size_t *ptr, std::size_t val) {
		// This pointer cast is probably UB.
		::atomicAdd((unsigned long long *)ptr, val);
	}
}

void histo_cuda(void *in_ptr, std::size_t size, void *out_ptr);

volatile auto histo_ptr = &histo_cuda;

using namespace std::literals::chrono_literals;

int main(int argc, char **argv) {
	if(argc != 3) {
		std::cerr << "Usage: histo <filename> <size>" << std::endl;
		std::abort();
	}
	std::size_t size = std::stoll(argv[2]);

	void *data;
	CUCH(cudaMallocManaged(&data, size));

	std::FILE *file = std::fopen(argv[1], "r");
	if(std::fread(data, 1, size, file) != size) {
		std::cerr << "Input error" << std::endl;
		std::abort();
	}
	std::fclose(file);

	std::size_t *counts;
	CUCH(cudaMallocManaged(&counts, NUM_VALUES * sizeof(std::size_t)));
	std::memset(counts, 0, NUM_VALUES * sizeof(std::size_t));

	auto t0 = std::chrono::steady_clock::now();
	histo_ptr(data, size, counts);
	auto t1 = std::chrono::steady_clock::now();
	std::fprintf(stderr, "%lu.%03u ms\n", (unsigned long) ((t1 - t0) / 1ms), (unsigned) ((t1 - t0) / 1us % 1000));

	CUCH(cudaFree(data));

	for(std::size_t v = 0; v < NUM_VALUES; v++) {
		std::printf("%12zu * 0x%02x", counts[v], (unsigned) v);
		if(v >= ' ' && v <= '~')
			std::printf(" ('%c')", (char) v);
		std::putchar('\n');
	}

	CUCH(cudaFree(counts));
	return 0;
}

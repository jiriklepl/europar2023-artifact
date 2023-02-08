#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <sstream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/traverser_iter.hpp>
#include <noarr/structures/interop/bag.hpp>

#ifdef CUDA
#include <noarr/structures/interop/cuda_traverser.cuh>

#define CUCH(status)  do { cudaError_t err = status; if (err != cudaSuccess) std::cerr << __FILE__ ":" << __LINE__ << ": error: " << cudaGetErrorString(err) << "\n\t" #status << std::endl, exit(err); } while (false)
#endif


using num_t = float;

template<typename A, typename B, typename C>
void matmul(A ta, B tb, C tc, char *pa, char *pb, char *pc);

using namespace std::literals::chrono_literals;

int main(int argc, char **argv) {
#ifdef MATRIX_SIZE
	if(argc != 2) {
		std::cerr << "Usage: PROGRAM FILE" << std::endl;
		std::abort();
	}

	auto i_st = noarr::array<'i', (std::size_t)MATRIX_SIZE>();
	auto j_st = noarr::array<'j', (std::size_t)MATRIX_SIZE>();
	auto k_st = noarr::array<'k', (std::size_t)MATRIX_SIZE>();
#else
	if(argc != 3) {
		std::cerr << "Usage: PROGRAM FILE SIZE" << std::endl;
		std::abort();
	}

	std::size_t size;

	{
		std::istringstream size_stream(argv[2]);
		size_stream >> size;
	}

	auto i_st = noarr::sized_vector<'i'>(size);
	auto j_st = noarr::sized_vector<'j'>(size);
	auto k_st = noarr::sized_vector<'k'>(size);
#endif

#ifdef A_ROW
	auto ta = noarr::scalar<num_t>() ^ i_st ^ j_st;
#else
#ifdef A_COL
	auto ta = noarr::scalar<num_t>() ^ j_st ^ i_st;
#else
#error define A_ROW or A_COL
#endif
#endif

#ifdef B_ROW
	auto tb = noarr::scalar<num_t>() ^ j_st ^ k_st;
#else
#ifdef B_COL
	auto tb = noarr::scalar<num_t>() ^ k_st ^ j_st;
#else
#error define B_ROW or B_COL
#endif
#endif

#ifdef C_ROW
	auto tc = noarr::scalar<num_t>() ^ i_st ^ k_st;
#else
#ifdef C_COL
	auto tc = noarr::scalar<num_t>() ^ k_st ^ i_st;
#else
#error define C_ROW or C_COL
#endif
#endif

	std::size_t a_sz = ta | noarr::get_size();
	std::size_t b_sz = tb | noarr::get_size();
	std::size_t c_sz = tc | noarr::get_size();

	char *data;

#ifdef CUDA
	CUCH(cudaMallocManaged(&data, a_sz + b_sz + c_sz));
#else
	if (!(data = (char *)malloc(a_sz + b_sz + c_sz))) {
		std::cerr << __FILE__ ":" << __LINE__ << ": error: failed to allocate memory" << std::endl;
		exit(1);
	}
#endif

	std::FILE *file = std::fopen(argv[1], "r");
	if(std::fread(data, 1, a_sz + b_sz, file) != a_sz + b_sz) {
		std::cerr << "Input error" << std::endl;
		std::abort();
	}
	std::fclose(file);

	matmul(ta, tb, tc, data, data + a_sz, data + a_sz + b_sz);

	auto t0 = std::chrono::steady_clock::now();
	matmul(ta, tb, tc, data, data + a_sz, data + a_sz + b_sz);
	auto t1 = std::chrono::steady_clock::now();
	std::fprintf(stderr, "%lu.%03u ms\n", (unsigned long) ((t1 - t0) / 1ms), (unsigned) ((t1 - t0) / 1us % 1000));

	std::fwrite(data + a_sz + b_sz, 1, c_sz, stdout);

#ifdef CUDA
	CUCH(cudaFree(data));
#else
	free(data);
#endif

	return 0;
}
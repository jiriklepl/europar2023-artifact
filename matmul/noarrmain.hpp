#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <sstream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>

#ifdef CUDA
#include <noarr/structures/interop/cuda_traverser.cuh>

#define CUCH(status)  do { cudaError_t err = status; if (err != cudaSuccess) std::cerr << __FILE__ ":" << __LINE__ << ": error: " << cudaGetErrorString(err) << "\n\t" #status << std::endl, exit(err); } while (false)
#endif

#ifdef LOGGING
#define LOG(log) \
	(std::cerr << log << std::endl)
#else
#define LOG(log) ((void)0)
#endif

using num_t = float;

#ifdef POLY
template<class F, std::size_t ...Idxs>
constexpr auto transform_pack(F f, std::index_sequence<Idxs...>) {
	return std::integer_sequence<
		typename decltype(f(std::integral_constant<std::size_t, 0>()))::value_type,
		decltype(f(std::integral_constant<std::size_t, Idxs>()))::value...>();
}

template<std::size_t I, std::size_t J, class C, C ...Idxs>
constexpr auto swap_pack(std::integer_sequence<C, Idxs...>) {
	constexpr std::size_t l = std::min(I, J);
	constexpr std::size_t h = std::max(I, J);
	constexpr C idxs[] = {Idxs...};

	return transform_pack([&]<std::size_t X>(std::integral_constant<std::size_t, X>) {
		if constexpr(X != l && X != h)
			return std::integral_constant<C, idxs[X]>();
		else if constexpr(X == l)
			return std::integral_constant<C, idxs[h]>();
		else
			return std::integral_constant<C, idxs[l]>();
	}, std::make_index_sequence<sizeof...(Idxs)>());
}
#endif

template<class A, class B, class C>
extern void run_matmul(A ta, B tb, C tc, num_t *pa, num_t *pb, num_t *pc);

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
	auto ta = noarr::scalar<num_t>() ^ i_st ^ k_st;
#else
#ifdef A_COL
	auto ta = noarr::scalar<num_t>() ^ k_st ^ i_st;
#else
#error define A_ROW or A_COL
#endif
#endif

#ifdef B_ROW
	auto tb = noarr::scalar<num_t>() ^ k_st ^ j_st;
#else
#ifdef B_COL
	auto tb = noarr::scalar<num_t>() ^ j_st ^ k_st;
#else
#error define B_ROW or B_COL
#endif
#endif

#ifdef C_ROW
	auto tc = noarr::scalar<num_t>() ^ i_st ^ j_st;
#else
#ifdef C_COL
	auto tc = noarr::scalar<num_t>() ^ j_st ^ i_st;
#else
#error define C_ROW or C_COL
#endif
#endif

	std::size_t a_sz = ta | noarr::get_size();
	std::size_t b_sz = tb | noarr::get_size();
	std::size_t c_sz = tc | noarr::get_size();

	num_t *data;

#ifdef CUDA
	CUCH(cudaMallocManaged(&data, a_sz + b_sz + c_sz));
#else
	if (!(data = (num_t *)malloc(a_sz + b_sz + c_sz))) {
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

	// run_matmul(ta, tb, tc, data, (data + a_sz / sizeof(num_t)), (data + (a_sz + b_sz) / sizeof(num_t)));

	auto start = std::chrono::high_resolution_clock::now();
	run_matmul(ta, tb, tc, data, (data + a_sz / sizeof(num_t)), (data + (a_sz + b_sz) / sizeof(num_t)));
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	std::cerr << duration.count() << std::endl;

	std::fwrite(data + (a_sz + b_sz) / sizeof(num_t), 1, c_sz, stdout);

#ifdef CUDA
	CUCH(cudaFree(data));
#else
	free(data);
#endif

	return 0;
}

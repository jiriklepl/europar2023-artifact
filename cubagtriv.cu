#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <cstring>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/cuda_traverser.cuh>

#define CUCH(status)  do { cudaError_t err = status; if (err != cudaSuccess) std::cerr << __FILE__ ":" << __LINE__ << ": error: " << cudaGetErrorString(err) << "\n\t" #status << std::endl, exit(err); } while (false)

using num_t = float;

template<typename T, typename A, typename B, typename C>
__global__ void kernel_matmul(T trav, A a, B b, C c) {
	num_t result = 0;

	trav.for_each([=, &result](auto ijk) {
		result += a[ijk] * b[ijk];
	});

	trav.order(noarr::reorder<>()).for_each([=, &result](auto ijk) {
		c[ijk] = result;
	});
}

template<typename A, typename B, typename C>
void matmul_cuda(A orig_a, B orig_b, C orig_c) {
	static constexpr auto I_BLOCK_SIZE = 32;
	static constexpr auto K_BLOCK_SIZE = 32;

	auto trav = noarr::cuda_traverser(a, b, c)
		.order(noarr::into_blocks_dynamic<'i', 'I', 'i', 'r'>(I_BLOCK_SIZE) ^ noarr::into_blocks_dynamic<'k', 'K', 'k', 's'>(K_BLOCK_SIZE))
		.template threads<'I', 'i', 'K', 'k'>();

	kernel_matmul<<<trav.grid_dim(), trav.block_dim()>>>(trav.inner(), a, b, c);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

using namespace std::literals::chrono_literals;

int main(int argc, char **argv) {
	if(argc != 2) {
		std::cerr << "Usage" << std::endl;
		std::abort();
	}

	auto i_st = noarr::array<'i', 8192>();
	auto j_st = noarr::array<'j', 8192>();
	auto k_st = noarr::array<'k', 8192>();

	auto ta = noarr::scalar<num_t>() ^ i_st ^ j_st;
	auto tb = noarr::scalar<num_t>() ^ j_st ^ k_st;
	auto tc = noarr::scalar<num_t>() ^ i_st ^ k_st;

	std::size_t a_sz = ta | noarr::get_size();
	std::size_t b_sz = tb | noarr::get_size();
	std::size_t c_sz = tc | noarr::get_size();

	char *data;
	CUCH(cudaMallocManaged(&data, a_sz + b_sz + c_sz));

	std::FILE *file = std::fopen(argv[1], "r");
	if(std::fread(data, 1, a_sz + b_sz, file) != a_sz + b_sz) {
		std::cerr << "Input error" << std::endl;
		std::abort();
	}
	std::fclose(file);

	auto a = noarr::make_bag(ta, data);
	auto b = noarr::make_bag(tb, data + a_sz);
	auto c = noarr::make_bag(tc, data + a_sz + b_sz);

	matmul_cuda(a, b, c);

	auto t0 = std::chrono::steady_clock::now();
	matmul_cuda(a, b, c);
	auto t1 = std::chrono::steady_clock::now();
	std::fprintf(stderr, "%lu.%03u ms\n", (unsigned long) ((t1 - t0) / 1ms), (unsigned) ((t1 - t0) / 1us % 1000));

	std::fwrite(data + a_sz + b_sz, 1, c_sz, stdout);

	CUCH(cudaFree(data));

	return 0;
}

#define CUDA
#include "histomain.hpp"

template<class Value, class Bin>
__global__ void kernel_histo(Value *in_ptr, Bin *out_ptr) {
	auto x = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * ELEMS_PER_THREAD;

	for (auto value = in_ptr + x; value < in_ptr + x + ELEMS_PER_THREAD; ++value)
		atomicAdd(&out_ptr[*value], 1);
}

void histo(void *in_ptr, std::size_t size, void *out_ptr) {
	auto block_dim = BLOCK_SIZE * ELEMS_PER_THREAD;
	auto grid_dim = (size - 1) / block_dim + 1;
	kernel_histo<<<grid_dim, BLOCK_SIZE>>>((value_t *)in_ptr, (std::size_t *)out_ptr);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

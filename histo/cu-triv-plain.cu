#define CUDA
#include "histomain.hpp"

template<class Value, class Bin>
__global__ void kernel_histo(Value *in_ptr, Bin *out_ptr) {
	auto start = blockIdx.x * ELEMS_PER_BLOCK + threadIdx.x;
	auto end = start + ELEMS_PER_BLOCK;

	for (auto i = start; i < end; i += BLOCK_SIZE) {
		Value value = in_ptr[i];
		atomicAdd(&out_ptr[value], 1);
	}
}

void histo(void *in_ptr, std::size_t size, void *out_ptr) {
	auto block_dim = BLOCK_SIZE;
	auto grid_dim = (size - 1) / ELEMS_PER_BLOCK + 1;

	kernel_histo<<<grid_dim, block_dim>>>((value_t *)in_ptr, (std::size_t *)out_ptr);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

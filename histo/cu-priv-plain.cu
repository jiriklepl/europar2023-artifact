#define CUDA
#include "histomain.hpp"

__global__ void histogram(value_t *in_ptr, std::size_t size, std::size_t *out_ptr) {
	extern __shared__ std::size_t shm_ptr[];

	auto start = blockIdx.x * ELEMS_PER_BLOCK + threadIdx.x;
	auto end = start + ELEMS_PER_BLOCK;
	auto my_copy_idx = threadIdx.x % NUM_COPIES;

	// Zero out shared memory. In this particular case, the access pattern happens
	for(auto i = threadIdx.x; i < NUM_VALUES * NUM_COPIES; i += BLOCK_SIZE) {
		shm_ptr[i] = 0;
	}

	__syncthreads();

	// Count the elements into the histogram copies in shared memory.
	for(auto i = start; i < end; i += BLOCK_SIZE) {
		auto value = in_ptr[i];
		atomicAdd(&shm_ptr[value * NUM_COPIES + my_copy_idx], 1);
	}

	__syncthreads();

	// Reduce the bins in shared memory into global memory.
	for(auto v = threadIdx.x; v < NUM_VALUES; v += BLOCK_SIZE) {
		std::size_t sum = 0;

		for(std::size_t i = 0; i < NUM_COPIES; i++) {
			auto collected_idx = (i + my_copy_idx) % NUM_COPIES;
			sum += shm_ptr[v * NUM_COPIES + collected_idx];
		}

		atomicAdd(&out_ptr[v], sum);
	}
}

void run_histogram(value_t *in_ptr, std::size_t size, std::size_t *out_ptr) {
	auto block_dim = BLOCK_SIZE;
	auto grid_dim = size / ELEMS_PER_BLOCK;
	auto shm_size = NUM_VALUES * NUM_COPIES * sizeof(std::size_t);

	histogram<<<grid_dim, block_dim, shm_size>>>((value_t *)in_ptr, size, (std::size_t *)out_ptr);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

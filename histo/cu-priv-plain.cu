#define CUDA
#include "histomain.hpp"

__global__ void kernel_histo(value_t *in_ptr, std::size_t size, std::size_t *out_ptr) {
	extern __shared__ std::size_t shm_ptr[];

    auto start = blockIdx.x * ELEMS_PER_BLOCK + threadIdx.x;
    auto end = (blockIdx.x + 1) * ELEMS_PER_BLOCK;
    auto my_copy_idx = threadIdx.x % NUM_COPIES;

	// Zero out shared memory. In this particular case, the access pattern happens
	for(auto i = threadIdx.x; i < NUM_VALUES * NUM_COPIES; i += BLOCK_SIZE) {
		shm_ptr[i] = 0;
	}

	__syncthreads();

	// Count the elements into the histogram copies in shared memory.
	for(auto i = start; i < end; i += BLOCK_SIZE) {
		auto value = in_ptr[i];
		atomicAdd(&shm_ptr[my_copy_idx * NUM_VALUES + value], 1);
	}

	__syncthreads();

	// Reduce the bins in shared memory into global memory.
	for(auto v = threadIdx.x; v < NUM_VALUES; v += BLOCK_SIZE) {
		std::size_t collected = 0;

		for(std::size_t i = 0; i < NUM_COPIES; i++) {
            auto collected_idx = (i + my_copy_idx) % NUM_COPIES;
			collected += shm_ptr[collected_idx * NUM_VALUES + v];
		}

		atomicAdd(&out_ptr[v], collected);
	}
}

void histo(void *in_ptr, std::size_t size, void *out_ptr) {
    auto blocks = (size - 1) / ELEMS_PER_BLOCK + 1;
    auto shm_size = NUM_VALUES * NUM_COPIES * sizeof(std::size_t);

    kernel_histo<<<blocks, BLOCK_SIZE, shm_size>>>((value_t *)in_ptr, size, (std::size_t *)out_ptr);

    CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}
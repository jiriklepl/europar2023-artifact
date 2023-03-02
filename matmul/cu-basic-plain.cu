#define CUDA
#include "plainmain.hpp"

#ifndef BLOCK_SIZE
#error define appropriate BLOCK_SIZE
#endif

template<class ISize, class JSize, class KSize>
__global__ void matmul(ISize i_size, JSize j_size, KSize k_size, num_t* glm_a, num_t* glm_b, num_t* glm_c) {
	num_t result = 0;

	auto i = blockIdx.x * blockDim.x + threadIdx.x;
	auto j = blockIdx.y * blockDim.y + threadIdx.y;

	for (std::size_t k = 0; k < k_size; k++) {
		result += glm_a[k*i_size + i] * glm_b[j*k_size + k];
	}

	glm_c[j*i_size + i] = result;
}

template<class ISize, class JSize, class KSize>
void run_matmul(ISize i_size, JSize j_size, KSize k_size, num_t* pa, num_t* pb, num_t* pc) {
	auto i_block_dim = uint(i_size / BLOCK_SIZE);
	auto j_block_dim = uint(j_size / BLOCK_SIZE);

	matmul<<<{i_block_dim, j_block_dim}, {(uint)BLOCK_SIZE, (uint)BLOCK_SIZE}>>>(i_size, j_size, k_size, pa, pb, pc);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

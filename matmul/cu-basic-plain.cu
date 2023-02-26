#define CUDA
#include "plainmain.hpp"

#ifndef BLOCK_SIZE
#error define appropriate BLOCK_SIZE
#endif

template<class ISize, class JSize, class KSize>
__global__ void kernel_matmul(ISize i_size, JSize j_size, KSize k_size, num_t* glm_a, num_t* glm_b, num_t* glm_c) {
	num_t result = 0;

	auto i = blockIdx.x * blockDim.x + threadIdx.x;
	auto k = blockIdx.y * blockDim.y + threadIdx.y;

	for (std::size_t j = 0; j < j_size; j++) {
		result += glm_a[j*i_size + i] * glm_b[k*j_size + j];
	}

	glm_c[k*i_size + i] = result;
}

template<class ISize, class JSize, class KSize>
void matmul(ISize i_size, JSize j_size, KSize k_size, num_t* cu_a, num_t* cu_b, num_t* cu_c) {
	auto i_block_dim = uint(i_size / BLOCK_SIZE);
	auto k_block_dim = uint(k_size / BLOCK_SIZE);

	kernel_matmul<<<{i_block_dim, k_block_dim}, {(uint)BLOCK_SIZE, (uint)BLOCK_SIZE}>>>(i_size, j_size, k_size, cu_a, cu_b, cu_c);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

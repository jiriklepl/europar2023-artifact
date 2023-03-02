#define CUDA
#include "policymain.hpp"

#ifndef BLOCK_SIZE
#error define appropriate BLOCK_SIZE
#endif

template<class ISize, class JSize, class KSize, class A, class B, class C>
__global__ void matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c) {
	num_t result = 0;

	auto i = blockIdx.x * blockDim.x + threadIdx.x;
	auto j = blockIdx.y * blockDim.y + threadIdx.y;

	for (std::size_t k = 0; k < k_size; k++) {
		result += a(k, i) * b(j, k);
	}

	c(j, i) = result;
}

template<class ISize, class JSize, class KSize, class A, class B, class C>
void run_matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c) {
	matmul<<<{(uint)(i_size/BLOCK_SIZE), (uint)(j_size/BLOCK_SIZE)}, {(uint)BLOCK_SIZE, (uint)BLOCK_SIZE}>>>(i_size, j_size, k_size, a, b, c);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

#define CUDA
#include "policymain.hpp"

#ifndef BLOCK_SIZE
#error define appropriate BLOCK_SIZE
#endif

template<class ISize, class JSize, class KSize, class A, class B, class C>
__global__ void kernel_matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c) {
	num_t result = 0;

	auto i = blockIdx.x * blockDim.x + threadIdx.x;
	auto k = blockIdx.y * blockDim.y + threadIdx.y;

	for (std::size_t j = 0; j < j_size; j++) {
		result += a(j, i) * b(k, j);
	}

	c(k, i) = result;
}

template<class ISize, class JSize, class KSize, class A, class B, class C>
void matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c) {

	kernel_matmul<<<{(uint)(i_size/BLOCK_SIZE), (uint)(k_size/BLOCK_SIZE)}, {(uint)BLOCK_SIZE, (uint)BLOCK_SIZE}>>>(i_size, j_size, k_size, a, b, c);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

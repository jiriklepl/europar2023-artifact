#define CUDA
#include "policymain.hpp"

static constexpr uint I_BLOCK_SIZE = 1024;
static constexpr uint K_BLOCK_SIZE = 8;

template<class ISize, class JSize, class KSize, class A, class B, class C>
__global__ void kernel_matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c) {
	extern __shared__ num_t shm_c[];
	auto d = make_matrix<ROW_MAJOR>(shm_c,  I_BLOCK_SIZE, K_BLOCK_SIZE);

	auto I = blockIdx.x * I_BLOCK_SIZE;
	auto K = blockIdx.y * K_BLOCK_SIZE;
	auto i = threadIdx.x;

	for(size_t k = 0; k < K_BLOCK_SIZE; k++) {
		d(k, i) = 0;
	}

	for(size_t j = 0; j < j_size; j++) {
		for(size_t k = 0; k < K_BLOCK_SIZE; k++) {
			d(k, i) += a(j, I + i) * b(K + k, j);
		}
	}

	for(size_t k = 0; k < K_BLOCK_SIZE; k++) {
		c(K + k, I + i) = d(k, i);
	}
}

template<class ISize, class JSize, class KSize, class A, class B, class C>
void matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c) {
	kernel_matmul<<<{i_size/I_BLOCK_SIZE, k_size/K_BLOCK_SIZE}, I_BLOCK_SIZE, I_BLOCK_SIZE * K_BLOCK_SIZE * sizeof(float)>>>(i_size, j_size, k_size, a, b, c);
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}
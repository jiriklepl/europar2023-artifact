#define CUDA
#include "policymain.hpp"

static constexpr uint I_BLOCK_SIZE = 1024;
static constexpr uint K_BLOCK_SIZE = 8;

template<class ISize, class JSize, class KSize, class A, class B, class C>
__global__ void kernel_matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c) {
	extern __shared__ num_t shm_c[];
	auto d = make_matrix<ROW_MAJOR>(shm_c,  K_BLOCK_SIZE, I_BLOCK_SIZE);

	auto I = blockIdx.x * I_BLOCK_SIZE;
	auto K = blockIdx.y * K_BLOCK_SIZE;
	auto i = threadIdx.x;

	for(std::size_t k = 0; k < K_BLOCK_SIZE; k++) {
		d(i, k) = 0;
	}

	for(std::size_t j = 0; j < j_size; j++) {
		for(std::size_t k = 0; k < K_BLOCK_SIZE; k++) {
			d(i, k) += a(j, I + i) * b(K + k, j);
		}
	}

	for(std::size_t k = 0; k < K_BLOCK_SIZE; k++) {
		c(K + k, I + i) = d(i, k);
	}
}

template<class ISize, class JSize, class KSize, class A, class B, class C>
void matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c) {
	kernel_matmul<<<{(uint)(i_size/I_BLOCK_SIZE), (uint)(k_size/K_BLOCK_SIZE)}, (uint)I_BLOCK_SIZE, (uint)(I_BLOCK_SIZE * K_BLOCK_SIZE * sizeof(num_t))>>>(i_size, j_size, k_size, a, b, c);
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

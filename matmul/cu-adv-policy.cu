#define CUDA
#include "policymain.hpp"

static constexpr uint I_BLOCK_SIZE = 1024;
static constexpr uint J_BLOCK_SIZE = 8;

template<class ISize, class JSize, class KSize, class A, class B, class C>
__global__ void matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c) {
	extern __shared__ num_t shm_c[];
	auto d = make_matrix<ROW_MAJOR>(shm_c,  J_BLOCK_SIZE, I_BLOCK_SIZE);

	auto I = blockIdx.x * I_BLOCK_SIZE;
	auto J = blockIdx.y * J_BLOCK_SIZE;
	auto i = threadIdx.x;

	for(std::size_t j = 0; j < J_BLOCK_SIZE; j++) {
		d(i, j) = 0;
	}

	for(std::size_t k = 0; k < k_size; k++) {
		for(std::size_t j = 0; j < J_BLOCK_SIZE; j++) {
			d(i, j) += a(k, I + i) * b(J + j, k);
		}
	}

	for(std::size_t j = 0; j < J_BLOCK_SIZE; j++) {
		c(J + j, I + i) = d(i, j);
	}
}

template<class ISize, class JSize, class KSize, class A, class B, class C>
void run_matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c) {
	matmul<<<{(uint)(i_size/I_BLOCK_SIZE), (uint)(j_size/J_BLOCK_SIZE)}, (uint)I_BLOCK_SIZE, (uint)(I_BLOCK_SIZE * J_BLOCK_SIZE * sizeof(num_t))>>>(i_size, j_size, k_size, a, b, c);
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

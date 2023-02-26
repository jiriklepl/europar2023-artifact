#define CUDA
#include "plainmain.hpp"

static constexpr uint I_BLOCK_SIZE = 1024;
static constexpr uint K_BLOCK_SIZE = 8;

template<class ISize, class JSize, class KSize>
__global__ void kernel_matmul(ISize i_size, JSize j_size, KSize k_size, num_t* glm_a, num_t* glm_b, num_t* glm_c) {
	extern __shared__ num_t shm_c[];

	auto I = blockIdx.x * I_BLOCK_SIZE;
	auto K = blockIdx.y * K_BLOCK_SIZE;
	auto i = threadIdx.x;

	for(std::size_t k = 0; k < K_BLOCK_SIZE; k++) {
		shm_c[k*I_BLOCK_SIZE + i] = 0;
	}

	for(std::size_t j = 0; j < j_size; j++) {
		for(std::size_t k = 0; k < K_BLOCK_SIZE; k++) {
			num_t local_a = glm_a[j*i_size + (I+i)];
			num_t local_b = glm_b[(K+k)*j_size + j];
			shm_c[k*I_BLOCK_SIZE + i] += local_a * local_b;
		}
	}

	for(std::size_t k = 0; k < K_BLOCK_SIZE; k++) {
		num_t local_c = shm_c[k*I_BLOCK_SIZE + i];
		glm_c[(K+k)*i_size + (I+i)] = local_c;
	}
}

template<class ISize, class JSize, class KSize>
void matmul(ISize i_size, JSize j_size, KSize k_size, num_t* cu_a, num_t* cu_b, num_t* cu_c) {
	kernel_matmul<<<{(uint)i_size/I_BLOCK_SIZE, (uint)k_size/K_BLOCK_SIZE}, (uint)I_BLOCK_SIZE, (uint)(K_BLOCK_SIZE * I_BLOCK_SIZE * sizeof(num_t))>>>(i_size, j_size, k_size, cu_a, cu_b, cu_c);
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

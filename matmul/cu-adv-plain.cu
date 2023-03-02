#define CUDA
#include "plainmain.hpp"

static constexpr uint I_BLOCK_SIZE = 1024;
static constexpr uint J_BLOCK_SIZE = 8;

template<class ISize, class JSize, class KSize>
__global__ void matmul(ISize i_size, JSize j_size, KSize k_size, num_t* glm_a, num_t* glm_b, num_t* glm_c) {
	extern __shared__ num_t shm_c[];

	auto I = blockIdx.x * I_BLOCK_SIZE;
	auto J = blockIdx.y * J_BLOCK_SIZE;
	auto i = threadIdx.x;

	for(std::size_t j = 0; j < J_BLOCK_SIZE; j++) {
		shm_c[i*J_BLOCK_SIZE + j] = 0;
	}

	for(std::size_t k = 0; k < k_size; k++) {
		for(std::size_t j = 0; j < J_BLOCK_SIZE; j++) {
			num_t local_a = glm_a[k*i_size + (I+i)];
			num_t local_b = glm_b[(J+j)*k_size + k];
			shm_c[i*J_BLOCK_SIZE + j] += local_a * local_b;
		}
	}

	for(std::size_t j = 0; j < J_BLOCK_SIZE; j++) {
		num_t local_c = shm_c[i*J_BLOCK_SIZE + j];
		glm_c[(J+j)*i_size + (I+i)] = local_c;
	}
}

template<class ISize, class JSize, class KSize>
void run_matmul(ISize i_size, JSize j_size, KSize k_size, num_t* pa, num_t* pb, num_t* pc) {
	matmul<<<{(uint)i_size/I_BLOCK_SIZE, (uint)j_size/J_BLOCK_SIZE}, (uint)I_BLOCK_SIZE, (uint)(J_BLOCK_SIZE * I_BLOCK_SIZE * sizeof(num_t))>>>(i_size, j_size, k_size, pa, pb, pc);
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

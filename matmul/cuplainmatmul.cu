#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <cstring>

#define CUCH(status)  do { cudaError_t err = status; if (err != cudaSuccess) std::cerr << __FILE__ ":" << __LINE__ << ": error: " << cudaGetErrorString(err) << "\n\t" #status << std::endl, exit(err); } while (false)

using num_t = float;

static constexpr std::size_t I_BLOCK_SIZE = 1024;
static constexpr std::size_t K_BLOCK_SIZE = 8;

template<std::size_t ISize, std::size_t JSize, std::size_t KSize>
__global__ void kernel_matmul(num_t* glm_a, num_t* glm_b, num_t* glm_c) {
	__shared__ num_t shm_c[K_BLOCK_SIZE*I_BLOCK_SIZE];

	auto I = blockIdx.x * I_BLOCK_SIZE;
	auto K = blockIdx.y * K_BLOCK_SIZE;
	auto i = threadIdx.x;

	for(size_t k = 0; k < K_BLOCK_SIZE; k++) {
		shm_c[k*I_BLOCK_SIZE + i] = 0;
	}

	for(size_t j = 0; j < JSize; j++) {
		for(size_t k = 0; k < K_BLOCK_SIZE; k++) {
			num_t local_a = glm_a[j*ISize + (I+i)];
			num_t local_b = glm_b[(K+k)*JSize + j];
			shm_c[k*I_BLOCK_SIZE + i] += local_a * local_b;
		}
	}

	for(size_t k = 0; k < K_BLOCK_SIZE; k++) {
		glm_c[(K+k)*ISize + (I+i)] = shm_c[k*I_BLOCK_SIZE + i];
	}
}

template<std::size_t ISize, std::size_t JSize, std::size_t KSize>
void matmul_cuda(num_t* cu_a, num_t* cu_b, num_t* cu_c) {
	kernel_matmul<ISize, JSize, KSize><<<{ISize/I_BLOCK_SIZE, KSize/K_BLOCK_SIZE}, I_BLOCK_SIZE>>>(cu_a, cu_b, cu_c);
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

using namespace std::literals::chrono_literals;

int main(int argc, char **argv) {
	if(argc != 2) {
		std::cerr << "Usage" << std::endl;
		std::abort();
	}

	constexpr std::size_t ISize = 8192;
	constexpr std::size_t JSize = 8192;
	constexpr std::size_t KSize = 8192;

	std::size_t a_cnt = ISize * JSize;
	std::size_t b_cnt = JSize * KSize;
	std::size_t c_cnt = ISize * KSize;

	std::size_t a_sz = a_cnt * sizeof(num_t);
	std::size_t b_sz = b_cnt * sizeof(num_t);
	std::size_t c_sz = c_cnt * sizeof(num_t);

	num_t *data;
	CUCH(cudaMallocManaged(&data, a_sz + b_sz + c_sz));

	std::FILE *file = std::fopen(argv[1], "r");
	if(std::fread(data, 1, a_sz + b_sz, file) != a_sz + b_sz) {
		std::cerr << "Input error" << std::endl;
		std::abort();
	}
	std::fclose(file);

	matmul_cuda<ISize, JSize, KSize>(data, data + a_cnt, data + a_cnt + b_cnt);

	auto t0 = std::chrono::steady_clock::now();
	matmul_cuda<ISize, JSize, KSize>(data, data + a_cnt, data + a_cnt + b_cnt);
	auto t1 = std::chrono::steady_clock::now();
	std::fprintf(stderr, "%lu.%03u ms\n", (unsigned long) ((t1 - t0) / 1ms), (unsigned) ((t1 - t0) / 1us % 1000));

	std::fwrite(data + a_cnt + b_cnt, 1, c_sz, stdout);

	CUCH(cudaFree(data));

	return 0;
}

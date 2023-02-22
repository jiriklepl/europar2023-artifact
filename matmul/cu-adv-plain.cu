#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <sstream>
#include <type_traits>

#define CUCH(status)  do { cudaError_t err = status; if (err != cudaSuccess) std::cerr << __FILE__ ":" << __LINE__ << ": error: " << cudaGetErrorString(err) << "\n\t" #status << std::endl, exit(err); } while (false)

using num_t = float;

static constexpr uint I_BLOCK_SIZE = 1024;
static constexpr uint K_BLOCK_SIZE = 8;

template<class ISize, class JSize, class KSize>
__global__ void kernel_matmul(ISize i_size, JSize j_size, KSize k_size, num_t* glm_a, num_t* glm_b, num_t* glm_c) {
	__shared__ num_t shm_c[K_BLOCK_SIZE*I_BLOCK_SIZE];

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
		glm_c[(K+k)*i_size + (I+i)] = shm_c[k*I_BLOCK_SIZE + i];
	}
}

template<class ISize, class JSize, class KSize>
void matmul_cuda(ISize i_size, JSize j_size, KSize k_size, num_t* cu_a, num_t* cu_b, num_t* cu_c) {
	kernel_matmul<<<{(uint)i_size/I_BLOCK_SIZE, (uint)k_size/K_BLOCK_SIZE}, I_BLOCK_SIZE>>>(i_size, j_size, k_size, cu_a, cu_b, cu_c);
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

using namespace std::literals::chrono_literals;

int main(int argc, char **argv) {
#ifdef MATRIX_SIZE
	if(argc != 2) {
		std::cerr << "Usage: PROGRAM FILE" << std::endl;
		std::abort();
	}

	constexpr auto ISize = std::integral_constant<std::size_t, MATRIX_SIZE>();
	constexpr auto JSize = std::integral_constant<std::size_t, MATRIX_SIZE>();
	constexpr auto KSize = std::integral_constant<std::size_t, MATRIX_SIZE>();
#else
	if(argc != 3) {
		std::cerr << "Usage: PROGRAM FILE SIZE" << std::endl;
		std::abort();
	}

	std::size_t size;

	{
		std::istringstream size_stream(argv[2]);
		size_stream >> size;
	}

	auto ISize = size;
	auto JSize = size;
	auto KSize = size;
#endif

	auto a_cnt = ISize * JSize;
	auto b_cnt = JSize * KSize;
	auto c_cnt = ISize * KSize;

	auto a_sz = a_cnt * sizeof(num_t);
	auto b_sz = b_cnt * sizeof(num_t);
	auto c_sz = c_cnt * sizeof(num_t);

	num_t *data;
	CUCH(cudaMallocManaged(&data, a_sz + b_sz + c_sz));

	std::FILE *file = std::fopen(argv[1], "r");
	if(std::fread(data, 1, a_sz + b_sz, file) != a_sz + b_sz) {
		std::cerr << "Input error" << std::endl;
		std::abort();
	}
	std::fclose(file);

	// matmul_cuda(ISize, JSize, KSize, data, data + a_cnt, data + a_cnt + b_cnt);

	auto start = std::chrono::high_resolution_clock::now();
	matmul_cuda(ISize, JSize, KSize, data, data + a_cnt, data + a_cnt + b_cnt);
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	std::cerr << duration.count() << std::endl;

	std::fwrite(data + a_cnt + b_cnt, 1, c_sz, stdout);

	CUCH(cudaFree(data));

	return 0;
}

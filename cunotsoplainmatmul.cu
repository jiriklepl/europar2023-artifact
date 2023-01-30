#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <cstring>

#define CUCH(status)  do { cudaError_t err = status; if (err != cudaSuccess) std::cerr << __FILE__ ":" << __LINE__ << ": error: " << cudaGetErrorString(err) << "\n\t" #status << std::endl, exit(err); } while (false)

using num_t = float;

static constexpr std::size_t I_BLOCK_SIZE = 1024;
static constexpr std::size_t K_BLOCK_SIZE = 8;

enum layout { ROW_MAJOR, COL_MAJOR };

template<class Scalar, class Step, layout Layout>
struct matrix {
	constexpr matrix(Scalar *data, Step row_count, Step col_count) noexcept
		: data(data), step((Layout == ROW_MAJOR) ? row_count : col_count) {
	}

	template<class Major, class Minor>
	constexpr Scalar &operator() (Major major, Minor minor) const noexcept {
		if constexpr (Layout == ROW_MAJOR)
			return data[major * step + minor];
		else
			return data[major * step + minor];
	}

	Scalar *data;
	Step step;
};

template<auto Layout, class Scalar, class Step>
constexpr auto make_matrix(Scalar *data, Step row_count, Step col_count) noexcept {
	return matrix<Scalar, Step, Layout>(data, row_count, col_count);
}

template<std::size_t ISize, std::size_t JSize, std::size_t KSize, class A, class B, class C>
__global__ void kernel_matmul(A a, B b, C c) {
	extern __shared__ num_t shm_c[];
	auto d = make_matrix<ROW_MAJOR>(shm_c,  I_BLOCK_SIZE, K_BLOCK_SIZE);

	auto I = blockIdx.x * I_BLOCK_SIZE;
	auto K = blockIdx.y * K_BLOCK_SIZE;
	auto i = threadIdx.x;

	for(size_t k = 0; k < K_BLOCK_SIZE; k++) {
		d(k, i) = 0;
	}

	for(size_t j = 0; j < JSize; j++) {
		for(size_t k = 0; k < K_BLOCK_SIZE; k++) {
			d(k, i) += a(j, I + i) * b(K + k, j);
		}
	}

	for(size_t k = 0; k < K_BLOCK_SIZE; k++) {
		c(K + k, I + i) = d(k, i);
	}
}

template<std::size_t ISize, std::size_t JSize, std::size_t KSize, class A, class B, class C>
void matmul_cuda(A a, B b, C c) {
	kernel_matmul<ISize, JSize, KSize><<<{ISize/I_BLOCK_SIZE, KSize/K_BLOCK_SIZE}, I_BLOCK_SIZE, I_BLOCK_SIZE * K_BLOCK_SIZE * sizeof(float)>>>(a, b, c);
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

	auto a = make_matrix<ROW_MAJOR>(
		data,
		std::integral_constant<std::size_t, ISize>(),
		std::integral_constant<std::size_t, JSize>());
	auto b = make_matrix<ROW_MAJOR>(
		data + a_cnt,
		std::integral_constant<std::size_t, JSize>(),
		std::integral_constant<std::size_t, KSize>());
	auto c = make_matrix<ROW_MAJOR>(
		data + a_cnt + b_cnt, 
		std::integral_constant<std::size_t, ISize>(),
		std::integral_constant<std::size_t, KSize>());

	matmul_cuda<ISize, JSize, KSize>(a, b, c);

	auto t0 = std::chrono::steady_clock::now();
	matmul_cuda<ISize, JSize, KSize>(a, b, c);
	auto t1 = std::chrono::steady_clock::now();
	std::fprintf(stderr, "%lu.%03u ms\n", (unsigned long) ((t1 - t0) / 1ms), (unsigned) ((t1 - t0) / 1us % 1000));

	std::fwrite(data + a_cnt + b_cnt, 1, c_sz, stdout);

	CUCH(cudaFree(data));

	return 0;
}

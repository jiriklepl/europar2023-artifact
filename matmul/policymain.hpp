#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <sstream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>

#ifdef CUDA
#include <noarr/structures/interop/cuda_traverser.cuh>

#define CUCH(status)  do { cudaError_t err = status; if (err != cudaSuccess) std::cerr << __FILE__ ":" << __LINE__ << ": error: " << cudaGetErrorString(err) << "\n\t" #status << std::endl, exit(err); } while (false)
#endif


using num_t = float;

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

using namespace std::literals::chrono_literals;

template<class ISize, class JSize, class KSize, class A, class B, class C>
void matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c);

int main(int argc, char **argv) {
#ifdef MATRIX_SIZE
	if(argc != 2) {
		std::cerr << "Usage: PROGRAM FILE" << std::endl;
		std::abort();
	}

	auto i_size = std::integral_constant<std::size_t, MATRIX_SIZE>();
	auto j_size = std::integral_constant<std::size_t, MATRIX_SIZE>();
	auto k_size = std::integral_constant<std::size_t, MATRIX_SIZE>();
#else
	if(argc != 3) {
		std::cerr << "Usage: PROGRAM FILE SIZE" << std::endl;
		std::abort();
	}

	uint size;

	{
		std::istringstream size_stream(argv[2]);
		size_stream >> size;
	}

	auto i_size = size;
	auto j_size = size;
	auto k_size = size;
#endif

	std::size_t a_cnt = i_size * j_size;
	std::size_t b_cnt = j_size * k_size;
	std::size_t c_cnt = i_size * k_size;

	std::size_t a_sz = a_cnt * sizeof(num_t);
	std::size_t b_sz = b_cnt * sizeof(num_t);
	std::size_t c_sz = c_cnt * sizeof(num_t);

	num_t *data;

#ifdef CUDA
	CUCH(cudaMallocManaged(&data, a_sz + b_sz + c_sz));
#else
	if (!(data = (num_t *)malloc(a_sz + b_sz + c_sz))) {
		std::cerr << __FILE__ ":" << __LINE__ << ": error: failed to allocate memory" << std::endl;
		exit(1);
	}
#endif

	std::FILE *file = std::fopen(argv[1], "r");
	if(std::fread(data, 1, a_sz + b_sz, file) != a_sz + b_sz) {
		std::cerr << "Input error" << std::endl;
		std::abort();
	}
	std::fclose(file);


#ifdef A_ROW
	auto a = make_matrix<ROW_MAJOR>(data, i_size, j_size);
#else
#ifdef A_COL
	auto a = make_matrix<COL_MAJOR>(data, i_size, j_size);
#else
#error define A_ROW or A_COL
#endif
#endif

#ifdef B_ROW
	auto b = make_matrix<ROW_MAJOR>(data + a_cnt, j_size, k_size);
#else
#ifdef B_COL
	auto b = make_matrix<COL_MAJOR>(data + a_cnt, j_size, k_size);
#else
#error define B_ROW or B_COL
#endif
#endif

#ifdef C_ROW
	auto c = make_matrix<ROW_MAJOR>(data + a_cnt + b_cnt, i_size, k_size);
#else
#ifdef C_COL
	auto c = make_matrix<COL_MAJOR>(data + a_cnt + b_cnt, i_size, k_size);
#else
#error define C_ROW or C_COL
#endif
#endif

	matmul(i_size, j_size, k_size, a, b, c);

	auto t0 = std::chrono::steady_clock::now();
	matmul(i_size, j_size, k_size, a, b, c);
	auto t1 = std::chrono::steady_clock::now();
	std::fprintf(stderr, "%lu.%03u ms\n", (unsigned long) ((t1 - t0) / 1ms), (unsigned) ((t1 - t0) / 1us % 1000));

	std::fwrite(data + a_cnt + b_cnt, 1, c_sz, stdout);

#ifdef CUDA
	CUCH(cudaFree(data));
#else
	free(data);
#endif

	return 0;
}

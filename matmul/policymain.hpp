#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <sstream>

#ifdef CUDA
#define CUCH(status)  do { cudaError_t err = status; if (err != cudaSuccess) std::cerr << __FILE__ ":" << __LINE__ << ": error: " << cudaGetErrorString(err) << "\n\t" #status << std::endl, exit(err); } while (false)
#else
#endif

#ifdef LOGGING
#define LOG(log) \
	(std::cerr << log << std::endl)
#else
#define LOG(log) ((void)0)
#endif

using num_t = float;

enum layout { ROW_MAJOR, COL_MAJOR };

template<class Pointer, class Step, layout Layout>
struct matrix {
	constexpr matrix(Pointer data, Step step) noexcept
		: data(data), step(step)
	{ }

	template<class Major, class Minor>
	constexpr decltype(auto) operator() (Major major, Minor minor) const noexcept {
		if constexpr (Layout == ROW_MAJOR)
			return data[major * step + minor];
		else
			return data[minor * step + major];
	}

	const Pointer data;
	const Step step;
};

template<layout Layout, class Pointer, class RowCount, class ColCount>
constexpr auto make_matrix(Pointer data, RowCount row_count, ColCount col_count) noexcept {
	if constexpr (Layout == ROW_MAJOR)
		return matrix<Pointer, RowCount, Layout>(data, row_count);
	else
		return matrix<Pointer, ColCount, Layout>(data, col_count);
}

template<class ISize, class JSize, class KSize, class A, class B, class C>
void run_matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c);

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

	std::size_t size;

	{
		std::istringstream size_stream(argv[2]);
		size_stream >> size;
	}

	auto i_size = size;
	auto j_size = size;
	auto k_size = size;
#endif

#ifdef A_ROW
#define A_LAYOUT ROW_MAJOR
#else
#ifdef A_COL
#define A_LAYOUT COL_MAJOR
#else
#error define A_ROW or A_COL
#endif
#endif

#ifdef B_ROW
#define B_LAYOUT ROW_MAJOR
#else
#ifdef B_COL
#define B_LAYOUT COL_MAJOR
#else
#error define B_ROW or B_COL
#endif
#endif

#ifdef C_ROW
#define C_LAYOUT ROW_MAJOR
#else
#ifdef C_COL
#define C_LAYOUT COL_MAJOR
#else
#error define C_ROW or C_COL
#endif
#endif

	auto a_cnt = i_size * k_size;
	auto b_cnt = k_size * j_size;
	auto c_cnt = i_size * j_size;

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

	auto a = make_matrix<A_LAYOUT>((const num_t *)data, i_size, k_size);
	auto b = make_matrix<B_LAYOUT>((const num_t *)data + a_cnt, k_size, j_size);
	auto c = make_matrix<C_LAYOUT>(data + a_cnt + b_cnt, i_size, j_size);

	// run_matmul(i_size, j_size, k_size, a, b, c);

	auto start = std::chrono::high_resolution_clock::now();
	run_matmul(i_size, j_size, k_size, a, b, c);
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	std::cerr << duration.count() << std::endl;

	std::fwrite(data + a_cnt + b_cnt, 1, c_sz, stdout);

#ifdef CUDA
	CUCH(cudaFree(data));
#else
	free(data);
#endif

	return 0;
}

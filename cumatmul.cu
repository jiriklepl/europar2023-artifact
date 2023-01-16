#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <cstring>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/cuda_traverser.cuh>

#define CUCH(status)  do { cudaError_t err = status; if (err != cudaSuccess) std::cerr << __FILE__ ":" << __LINE__ << ": error: " << cudaGetErrorString(err) << "\n\t" #status << std::endl, exit(err); } while (false)

using num_t = float;

template<typename T, typename TA, typename TB, typename TC, typename TD>
__global__ void kernel_matmul(T trav, TA ta, TB tb, TC tc, TD td, void *pa, void *pb, void *pc) {
	extern __shared__ char pd[];

	trav.order(noarr::reorder<'k'>()).for_each([=](auto sa, auto sb, auto sc, auto sd) {
		td | noarr::get_at(pd, sd) = 0;
	});

	trav.order(noarr::reorder<'j', 'k'>()).for_each([=](auto sa, auto sb, auto sc, auto sd) {
		num_t a_elem = ta | noarr::get_at(pa, sa);
		num_t b_elem = tb | noarr::get_at(pb, sb);
		td | noarr::get_at(pd, sd) += a_elem * b_elem;
	});

	trav.order(noarr::reorder<'k'>()).for_each([=](auto sa, auto sb, auto sc, auto sd) {
		num_t c_elem = td | noarr::get_at(pd, sd);
		tc | noarr::get_at(pc, sc) = c_elem;
	});
}

template<typename A, typename B, typename C>
void matmul_cuda(A orig_ta, B orig_tb, C orig_tc, void *pa, void *pb, void *pc) {
	auto i_blocks = noarr::into_blocks<'i', /*'r',*/ 'I', 'i'>(noarr::idx<1024>);
	auto k_blocks = noarr::into_blocks<'k', /*'s',*/ 'K', 'k'>(noarr::idx<8>);

	auto ta = orig_ta ^ i_blocks;
	auto tb = orig_tb ^ k_blocks;
	auto tc = orig_tc ^ i_blocks ^ k_blocks;

	noarr::traverser(tc).order(noarr::reorder</*'s', 'r'*/>()).for_each([=](auto rs) {
		auto td = noarr::scalar<num_t>()
			^ noarr::sized_vector<'i'>(tc | noarr::get_length<'i'>(rs))
			^ noarr::sized_vector<'k'>(tc | noarr::get_length<'k'>(rs));

		auto trav = noarr::cuda_traverser(ta, tb, tc, td).order(noarr::fix(rs) ^ noarr::bcast<'1'>(1)).template threads<'I', 'i', 'K', '1'>();
		kernel_matmul<<<trav.grid_dim(), trav.block_dim(), td | noarr::get_size()>>>(trav.inner(), ta, tb, tc, td, pa, pb, pc);

		CUCH(cudaGetLastError());
	});

	CUCH(cudaDeviceSynchronize());
}

using namespace std::literals::chrono_literals;

int main(int argc, char **argv) {
	if(argc != 2) {
		std::cerr << "Usage" << std::endl;
		std::abort();
	}

	auto i_st = noarr::array<'i', 8192>();
	auto j_st = noarr::array<'j', 8192>();
	auto k_st = noarr::array<'k', 8192>();

	auto ta = noarr::scalar<num_t>() ^ i_st ^ j_st;
	auto tb = noarr::scalar<num_t>() ^ j_st ^ k_st;
	auto tc = noarr::scalar<num_t>() ^ i_st ^ k_st;

	std::size_t a_sz = ta | noarr::get_size();
	std::size_t b_sz = tb | noarr::get_size();
	std::size_t c_sz = tc | noarr::get_size();

	char *data;
	CUCH(cudaMallocManaged(&data, a_sz + b_sz + c_sz));

	std::FILE *file = std::fopen(argv[1], "r");
	if(std::fread(data, 1, a_sz + b_sz, file) != a_sz + b_sz) {
		std::cerr << "Input error" << std::endl;
		std::abort();
	}
	std::fclose(file);

	matmul_cuda(ta, tb, tc, data, data + a_sz, data + a_sz + b_sz);

	auto t0 = std::chrono::steady_clock::now();
	matmul_cuda(ta, tb, tc, data, data + a_sz, data + a_sz + b_sz);
	auto t1 = std::chrono::steady_clock::now();
	std::fprintf(stderr, "%lu.%03u ms\n", (unsigned long) ((t1 - t0) / 1ms), (unsigned) ((t1 - t0) / 1us % 1000));

	std::fwrite(data + a_sz + b_sz, 1, c_sz, stdout);

	CUCH(cudaFree(data));

	return 0;
}

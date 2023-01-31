#define CUDA
#include "matmulmain.hpp"

template<typename T, typename TA, typename TB, typename TC>
__global__ void kernel_matmul(T trav, TA ta, TB tb, TC tc, void *pa, void *pb, void *pc) {
	num_t result = 0;

	trav.for_each([=, &result](auto ijk) {
		num_t a_elem = ta | noarr::get_at(pa, ijk);
		num_t b_elem = tb | noarr::get_at(pb, ijk);
		result += a_elem * b_elem;
	});

	tc | noarr::get_at(pc, trav.state()) = result;
}

template<typename A, typename B, typename C>
void matmul(A orig_ta, B orig_tb, C orig_tc, char *pa, char *pb, char *pc) {
	auto i_blocks = noarr::into_blocks<'i', /*'r',*/ 'I', 'i'>(noarr::lit<32>);
	auto k_blocks = noarr::into_blocks<'k', /*'s',*/ 'K', 'k'>(noarr::lit<32>);

	auto ta = orig_ta ^ i_blocks;
	auto tb = orig_tb ^ k_blocks;
	auto tc = orig_tc ^ i_blocks ^ k_blocks;

	auto trav = noarr::cuda_traverser(ta, tb, tc).template threads<'I', 'i', 'K', 'k'>();
	kernel_matmul<<<trav.grid_dim(), trav.block_dim()>>>(trav.inner(), ta, tb, tc, pa, pb, pc);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

#define CUDA
#include "matmulmain.hpp"

#ifndef BLOCK_SIZE
#error define appropriate BLOCK_SIZE
#endif

template<typename T, typename TA, typename TB, typename TC>
__global__ void kernel_matmul(T trav, TA ta, TB tb, TC tc, void *pa, void *pb, void *pc) {
	trav.template for_dims<'r', 's'>([=](auto trav) {
		num_t result = 0;

		trav.for_each([=, &result](auto ijk) {
			num_t a_elem = ta | noarr::get_at(pa, ijk);
			num_t b_elem = tb | noarr::get_at(pb, ijk);
			result += a_elem * b_elem;
		});

		tc | noarr::get_at(pc, trav.state()) = result;
	});
}

template<typename A, typename B, typename C>
void matmul(A ta, B tb, C tc, char *pa, char *pb, char *pc) {
#ifdef DYNAMIC_BLOCKS
	auto into_blocks = noarr::into_blocks_dynamic<'i', 'I', 'i', 'r'>(noarr::lit<BLOCK_SIZE>) ^ noarr::into_blocks_dynamic<'k', 'K', 'k', 's'>(noarr::lit<BLOCK_SIZE>);
#else
	auto into_blocks = noarr::into_blocks<'i', 'I', 'i'>(noarr::lit<BLOCK_SIZE>) ^ noarr::into_blocks<'k', 'K', 'k'>(noarr::lit<BLOCK_SIZE>) ^ noarr::bcast<'r'>(noarr::lit<1>) ^ noarr::bcast<'s'>(noarr::lit<1>);
#endif

	auto trav = noarr::cuda_threads<'I', 'i', 'K', 'k'>(noarr::traverser(ta, tb, tc).order(into_blocks));
	kernel_matmul<<<trav.grid_dim(), trav.block_dim()>>>(trav.inner(), ta, tb, tc, pa, pb, pc);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

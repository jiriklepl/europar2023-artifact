#define CUDA
#include "noarrmain.hpp"

#ifndef BLOCK_SIZE
#error define appropriate BLOCK_SIZE
#endif

template<class T, class TA, class TB, class TC>
__global__ void kernel_matmul(T trav, TA ta, TB tb, TC tc, num_t *pa, num_t *pb, num_t *pc) {
	using noarr::get_at;

	trav.template for_dims<'r', 's'>([=](auto trav) {
		num_t result = 0;

		trav.template for_dims<'j'>([=, &result](auto ijk) {
			result += (ta | get_at(pa, ijk)) * (tb | get_at(pb, ijk));
		});

		tc | get_at(pc, trav.state()) = result;
	});
}

template<class A, class B, class C>
void matmul(A ta, B tb, C tc, num_t *pa, num_t *pb, num_t *pc) {
#ifdef DYNAMIC_BLOCKS
	auto into_blocks = noarr::into_blocks_dynamic<'i', 'I', 'i', 'r'>(noarr::lit<BLOCK_SIZE>) ^ noarr::into_blocks_dynamic<'k', 'K', 'k', 's'>(noarr::lit<BLOCK_SIZE>);
#else
	auto into_blocks = noarr::into_blocks<'i', 'I', 'i'>(noarr::lit<BLOCK_SIZE>) ^ noarr::into_blocks<'k', 'K', 'k'>(noarr::lit<BLOCK_SIZE>) ^ noarr::bcast<'r', 's'>(noarr::lit<1>, noarr::lit<1>);
#endif

	auto cutrav = noarr::cuda_threads<'I', 'i', 'K', 'k'>(noarr::traverser(ta, tb, tc).order(into_blocks));

	cutrav.simple_run(kernel_matmul, 0, ta, tb, tc, pa, pb, pc);
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

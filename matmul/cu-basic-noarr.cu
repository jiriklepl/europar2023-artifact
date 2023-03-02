#define CUDA
#include "noarrmain.hpp"

#ifndef BLOCK_SIZE
#error define appropriate BLOCK_SIZE
#endif

template<class T, class TA, class TB, class TC>
__global__ void matmul(T trav, TA ta, TB tb, TC tc, num_t *pa, num_t *pb, num_t *pc) {
	trav.template for_dims<'r', 's'>([=](auto trav) {
		num_t result = 0;

		trav.template for_dims<'k'>([=, &result](auto inner) {
			auto state = inner.state();
			result += (ta | noarr::get_at(pa, state)) * (tb | noarr::get_at(pb, state));
		});

		tc | noarr::get_at(pc, trav.state()) = result;
	});
}

template<class A, class B, class C>
void run_matmul(A ta, B tb, C tc, num_t *pa, num_t *pb, num_t *pc) {
#ifdef DYNAMIC_BLOCKS
	auto into_blocks = noarr::into_blocks_dynamic<'i', 'I', 'i', 'r'>(noarr::lit<BLOCK_SIZE>) ^ noarr::into_blocks_dynamic<'j', 'J', 'j', 's'>(noarr::lit<BLOCK_SIZE>);
#else
	auto into_blocks = noarr::into_blocks<'i', 'I', 'i'>(noarr::lit<BLOCK_SIZE>) ^ noarr::into_blocks<'j', 'J', 'j'>(noarr::lit<BLOCK_SIZE>) ^ noarr::bcast<'r', 's'>(noarr::lit<1>, noarr::lit<1>);
#endif

	auto cutrav = noarr::cuda_threads<'I', 'i', 'J', 'j'>(noarr::traverser(ta, tb, tc).order(into_blocks));

	cutrav.simple_run(matmul, 0, ta, tb, tc, pa, pb, pc);
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

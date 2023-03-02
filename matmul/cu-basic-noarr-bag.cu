#define CUDA
#include "noarrmain.hpp"

#include <noarr/structures/interop/bag.hpp>

#ifndef BLOCK_SIZE
#error define appropriate BLOCK_SIZE
#endif

template<class T, class A, class B, class C>
__global__ void matmul(T trav, A a, B b, C c) {
	trav.template for_dims<'r', 's'>([=](auto trav) {
		num_t result = 0;

		trav.template for_dims<'k'>([=, &result](auto inner) {
			auto state = inner.state();
			result += a[state] * b[state];
		});

		c[trav.state()] = result;
	});
}

template<class A, class B, class C>
void run_matmul(A ta, B tb, C tc, num_t *pa, num_t *pb, num_t *pc) {
	auto a = noarr::make_bag(ta, pa);
	auto b = noarr::make_bag(tb, pb);
	auto c = noarr::make_bag(tc, pc);

#ifdef DYNAMIC_BLOCKS
	auto into_blocks = noarr::into_blocks_dynamic<'i', 'I', 'i', 'r'>(noarr::lit<BLOCK_SIZE>) ^ noarr::into_blocks_dynamic<'j', 'J', 'j', 's'>(noarr::lit<BLOCK_SIZE>);
#else
	auto into_blocks = noarr::into_blocks<'i', 'I', 'i'>(noarr::lit<BLOCK_SIZE>) ^ noarr::into_blocks<'j', 'J', 'j'>(noarr::lit<BLOCK_SIZE>) ^ noarr::bcast<'r', 's'>(noarr::lit<1>, noarr::lit<1>);
#endif

	auto cutrav = noarr::cuda_threads<'I', 'i', 'J', 'j'>(noarr::traverser(a, b, c).order(into_blocks));

	cutrav.simple_run(matmul, 0, a, b, c);
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

#define CUDA
#include "../matmul/noarrmain.hpp"

#ifndef BLOCK_SIZE
#error define appropriate BLOCK_SIZE
#endif

template<class T, class A, class B, class C>
__global__ void matmul(T trav, A a, B b, C c) {
	num_t result = 0;

	trav.for_each([=](auto j) {
		result += a[j] * b[j];
	});

	c[trav.state()] = result;
}

template<class A, class B, class C>
void run_matmul(A ta, B tb, C tc, num_t *pa, num_t *pb, num_t *pc) {
	auto a = noarr::make_bag(ta, pa);
	auto b = noarr::make_bag(tb, pb);
	auto c = noarr::make_bag(tc, pc);

	auto into_blocks = noarr::into_blocks<'i', 'I', 'i'>(noarr::lit<BLOCK_SIZE>)
		^ noarr::into_blocks<'j', 'J', 'j'>(noarr::lit<BLOCK_SIZE>);

	auto cutrav = noarr::cuda_threads<'I', 'i', 'J', 'j'>(noarr::traverser(a, b, c).order(into_blocks));

	cutrav.simple_run(matmul, 0, a, b, c);
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

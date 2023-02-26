#define CUDA
#include "../matmul/noarrmain.hpp"

#ifndef BLOCK_SIZE
#error define appropriate BLOCK_SIZE
#endif

template<class T, class A, class B, class C>
__global__ void kernel_matmul(T trav, A a, B b, C c) {
	num_t result = 0;

	trav.for_each([=](auto j) {
		result += a[j] * b[j];
	});

	c[trav.state()] = result;
}

template<class A, class B, class C>
void matmul(A ta, B tb, C tc, char *pa, char *pb, char *pc) {
	auto a = noarr::make_bag(ta, pa);
	auto b = noarr::make_bag(tb, pb);
	auto c = noarr::make_bag(tc, pc);

	auto into_blocks = noarr::into_blocks<'i', 'I', 'i'>(noarr::lit<BLOCK_SIZE>)
		^ noarr::into_blocks<'k', 'K', 'k'>(noarr::lit<BLOCK_SIZE>);

	auto cutrav = noarr::cuda_threads<'I', 'i', 'K', 'k'>(noarr::traverser(a, b, c).order(into_blocks));

	kernel_matmul<<<cutrav.grid_dim(), cutrav.block_dim()>>>(cutrav.inner(), a, b, c);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

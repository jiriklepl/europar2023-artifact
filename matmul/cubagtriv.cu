#define CUDA
#include "matmulmain.hpp"

#ifndef BLOCK_SIZE
#error define appropriate BLOCK_SIZE
#endif

template<typename T, typename A, typename B, typename C>
__global__ void kernel_matmul(T trav, A a, B b, C c) {
	trav.template for_dims<'r', 's'>([=](auto trav) {
    	num_t result = 0;

        trav.for_each([&](auto j) {
            result += a[j] * b[j];
        });

        c[trav.state()] = result;
    });
}

template<typename A, typename B, typename C>
void matmul(A ta, B tb, C tc, char *pa, char *pb, char *pc) {
	auto a = noarr::make_bag(ta, pa);
	auto b = noarr::make_bag(tb, pb);
	auto c = noarr::make_bag(tc, pc);

#ifdef DYNAMIC_BLOCKS
	auto into_blocks = noarr::into_blocks_dynamic<'i', 'I', 'i', 'r'>(noarr::lit<BLOCK_SIZE>) ^ noarr::into_blocks_dynamic<'k', 'K', 'k', 's'>(noarr::lit<BLOCK_SIZE>);
#else
	auto into_blocks = noarr::into_blocks<'i', 'I', 'i'>(noarr::lit<BLOCK_SIZE>) ^ noarr::into_blocks<'k', 'K', 'k'>(noarr::lit<BLOCK_SIZE>) ^ noarr::bcast<'r'>(noarr::lit<1>) ^ noarr::bcast<'s'>(noarr::lit<1>);
#endif

	auto trav = noarr::cuda_threads<'I', 'i', 'K', 'k'>(noarr::traverser(a, b, c).order(into_blocks));

	kernel_matmul<<<trav.grid_dim(), trav.block_dim()>>>(trav.inner(), a, b, c);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}
#define CUDA
#include "matmulmain.hpp"

#ifndef BLOCK_SIZE
#error define appropriate BLOCK_SIZE
#endif

template<typename T, typename A, typename B, typename C>
__global__ void kernel_matmul(T trav, A a, B b, C c) {
	trav.for_dims<'r', 's'>([=](auto inner) {
    	num_t result = 0;

        inner.for_each([&](auto j) {
            result += a[j] * b[j];
        });

        c[inner.state()] = result;
    });
}

template<typename A, typename B, typename C>
void matmul(A ta, B tb, C tc, char *pa, char *pb, char *pc) {
	auto a = noarr::make_bag(ta, pa);
	auto b = noarr::make_bag(tb, pb);
	auto c = noarr::make_bag(tc, pc);

	auto trav = noarr::cuda_traverser(a, b, c)
		.order(noarr::into_blocks_dynamic<'i', 'I', 'i', 'r'>(BLOCK_SIZE) ^ noarr::into_blocks_dynamic<'k', 'K', 'k', 's'>(BLOCK_SIZE))
		.template threads<'I', 'i', 'K', 'k'>();

	kernel_matmul<<<trav.grid_dim(), trav.block_dim()>>>(trav.inner(), a, b, c);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

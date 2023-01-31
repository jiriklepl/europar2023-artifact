#define CUDA
#include "matmulmain.hpp"

template<typename T, typename C>
__global__ void kernel_bzero(T trav, C c) {
	trav.for_each([&](auto state) {
		c[state] = 0;
	});
}

template<typename T, typename A, typename B, typename C>
__global__ void kernel_matmul(T trav, A a, B b, C c) {
	trav.for_each([&](auto state) {
		c[state] += a[state] * b[state];
	});
}

template<typename A, typename B, typename C>
void matmul(A ta, B tb, C tc, char *pa, char *pb, char *pc) {
	auto a = noarr::make_bag(ta, pa);
	auto b = noarr::make_bag(tb, pb);
	auto c = noarr::make_bag(tc, pc);

	static constexpr auto I_BLOCK_SIZE = 32;
	static constexpr auto K_BLOCK_SIZE = 32;

	{
		auto trav = noarr::cuda_traverser(c)
			.order(noarr::into_blocks_dynamic<'i', 'I', 'i', 'r'>(I_BLOCK_SIZE) ^ noarr::into_blocks_dynamic<'k', 'K', 'k', 's'>(K_BLOCK_SIZE))
			.template threads<'I', 'i', 'K', 'k'>();

		kernel_bzero<<<trav.grid_dim(), trav.block_dim()>>>(trav.inner(), c);
		CUCH(cudaGetLastError());
	}

	{
		auto trav = noarr::cuda_traverser(a, b, c)
			.order(noarr::hoist<'i'>() ^ noarr::hoist<'k'>())
			.order(noarr::into_blocks_dynamic<'i', 'I', 'i', 'r'>(I_BLOCK_SIZE) ^ noarr::into_blocks_dynamic<'k', 'K', 'k', 's'>(K_BLOCK_SIZE))
			.template threads<'I', 'i', 'K', 'k'>();

		kernel_matmul<<<trav.grid_dim(), trav.block_dim()>>>(trav.inner(), a, b, c);
		CUCH(cudaGetLastError());
	}

	CUCH(cudaDeviceSynchronize());
}

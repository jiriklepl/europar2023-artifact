#define CUDA
#include "noarrmain.hpp"

#include <noarr/structures/interop/bag.hpp>

template<class T, class A, class B, class C, class TD>
__global__ void kernel_matmul(T trav, A a, B b, C c, TD td) {
	extern __shared__ char pd[];
	auto d = noarr::make_bag(td, pd);

	trav.template for_dims<'k'>([=](auto inner) {
		d[inner.state()] = 0;
	});

	trav.template for_dims<'j', 'k'>([=](auto inner) {
		auto &&ijk = inner.state();
		d[ijk] += a[ijk] * b[ijk];
	});

	trav.template for_dims<'k'>([=](auto inner) {
		auto &&ik = inner.state();
		c[ik] = d[ik];
	});
}

template<class A, class B, class C>
void matmul(A ta, B tb, C tc, num_t *pa, num_t *pb, num_t *pc) {
	auto i_blocks = noarr::into_blocks<'i', 'I', 'i'>(noarr::lit<1024>);
	auto k_blocks = noarr::into_blocks<'k', 'K', 'k'>(noarr::lit<8>);

	auto a = noarr::make_bag(ta ^ i_blocks, pa);
	auto b = noarr::make_bag(tb ^ k_blocks, pb);
	auto c = noarr::make_bag(tc ^ i_blocks ^ k_blocks, pc);

	auto trav = noarr::traverser(a, b, c).order(noarr::bcast<'1'>(1));
	auto td = noarr::scalar<num_t>() ^ noarr::vectors_like<'k', 'i'>(trav.top_struct());

	auto cutrav = noarr::cuda_threads<'I', 'i', 'K', '1'>(trav);
	kernel_matmul<<<cutrav.grid_dim(), cutrav.block_dim(), td | noarr::get_size()>>>(cutrav.inner(), a, b, c, td);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

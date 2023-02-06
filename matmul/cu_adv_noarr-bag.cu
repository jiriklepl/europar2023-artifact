#define CUDA
#include "noarrmain.hpp"

template<typename T, typename A, typename B, typename C, typename TD>
__global__ void kernel_matmul(T trav, A a, B b, C c, TD td) {
	extern __shared__ char pd[];
	auto d = noarr::make_bag(td, pd);

	trav.template for_dims<'k'>([&](auto inner) {
		d[inner.state()] = 0;
	});

	trav.template for_dims<'j', 'k'>([&](auto inner) {
		auto ijk = inner.state();
		d[ijk] += a[ijk] * b[ijk];
	});

	trav.template for_dims<'k'>([&](auto inner) {
		auto ik = inner.state();
		c[ik] = d[ik];
	});
}

template<typename A, typename B, typename C>
void matmul(A orig_ta, B orig_tb, C orig_tc, char *pa, char *pb, char *pc) {
	auto i_blocks = noarr::into_blocks<'i', /*'r',*/ 'I', 'i'>(noarr::lit<1024>);
	auto k_blocks = noarr::into_blocks<'k', /*'s',*/ 'K', 'k'>(noarr::lit<8>);

	auto a = noarr::make_bag(orig_ta ^ i_blocks, pa);
	auto b = noarr::make_bag(orig_tb ^ k_blocks, pb);
	auto c = noarr::make_bag(orig_tc ^ i_blocks ^ k_blocks, pc);

	noarr::traverser(a, b, c).template for_dims</*'s', 'r'*/>([=](auto trav) {
		auto td = noarr::scalar<num_t>()
			^ noarr::sized_vector<'i'>(c.template get_length<'i'>(trav.state()))
			^ noarr::sized_vector<'k'>(c.template get_length<'k'>(trav.state()));

		auto cutrav = noarr::cuda_threads<'I', 'i', 'K', '1'>(trav.order(noarr::bcast<'1'>(1)));
		kernel_matmul<<<cutrav.grid_dim(), cutrav.block_dim(), td | noarr::get_size()>>>(cutrav.inner(), a, b, c, td);

		CUCH(cudaGetLastError());
	});

	CUCH(cudaDeviceSynchronize());
}

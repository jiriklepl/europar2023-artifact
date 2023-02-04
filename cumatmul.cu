#define CUDA
#include "matmulmain.hpp"

template<typename T, typename TA, typename TB, typename TC, typename TD>
__global__ void kernel_matmul(T trav, TA ta, TB tb, TC tc, TD td, void *pa, void *pb, void *pc) {
	extern __shared__ char pd[];

	trav.template for_dims<'k'>([&](auto inner) {
		td | noarr::get_at(pd, inner.state()) = 0;
	});

	trav.template for_dims<'j', 'k'>([&](auto inner) {
		auto ijk = inner.state();
		num_t a_elem = ta | noarr::get_at(pa, ijk);
		num_t b_elem = tb | noarr::get_at(pb, ijk);
		td | noarr::get_at(pd, ijk) += a_elem * b_elem;
	});

	trav.template for_dims<'k'>([&](auto inner) {
		auto ik = inner.state();
		num_t c_elem = td | noarr::get_at(pd, ik);
		tc | noarr::get_at(pc, ik) = c_elem;
	});
}

template<typename A, typename B, typename C>
void matmul(A orig_ta, B orig_tb, C orig_tc, char *pa, char *pb, char *pc) {
	auto i_blocks = noarr::into_blocks<'i', /*'r',*/ 'I', 'i'>(noarr::lit<1024>);
	auto k_blocks = noarr::into_blocks<'k', /*'s',*/ 'K', 'k'>(noarr::lit<8>);

	auto ta = orig_ta ^ i_blocks;
	auto tb = orig_tb ^ k_blocks;
	auto tc = orig_tc ^ i_blocks ^ k_blocks;

	noarr::traverser(ta, tb, tc).template for_dims</*'s', 'r'*/>([=](auto trav) {
		auto td = noarr::scalar<num_t>()
			^ noarr::sized_vector<'i'>(tc | noarr::get_length<'i'>(trav.state()))
			^ noarr::sized_vector<'k'>(tc | noarr::get_length<'k'>(trav.state()));

		auto cutrav = noarr::cuda_threads<'I', 'i', 'K', '1'>(trav.order(noarr::bcast<'1'>(1)));
		kernel_matmul<<<cutrav.grid_dim(), cutrav.block_dim(), td | noarr::get_size()>>>(cutrav.inner(), ta, tb, tc, td, pa, pb, pc);

		CUCH(cudaGetLastError());
	});

	CUCH(cudaDeviceSynchronize());
}

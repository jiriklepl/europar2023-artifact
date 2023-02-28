#define CUDA
#include "noarrmain.hpp"

template<class T, class TA, class TB, class TC, class TD>
__global__ void kernel_matmul(T trav, TA ta, TB tb, TC tc, TD td, num_t *pa, num_t *pb, num_t *pc) {
	extern __shared__ char pd[];

	trav.template for_dims<'k'>([=](auto inner) {
		td | noarr::get_at(pd, inner.state()) = 0;
	});

	trav.template for_dims<'j', 'k'>([=](auto inner) {
		auto at = noarr::getter(inner.state());
		td | at(pd) += (ta | at(pa)) * (tb | at(pb));
	});

	trav.template for_dims<'k'>([=](auto inner) {
		auto at = noarr::getter(inner.state());
		tc | at(pc) = td | at(pd);
	});
}

template<class A, class B, class C>
void matmul(A orig_ta, B orig_tb, C orig_tc, num_t *pa, num_t *pb, num_t *pc) {
	auto i_blocks = noarr::into_blocks<'i', 'I', 'i'>(noarr::lit<1024>);
	auto k_blocks = noarr::into_blocks<'k', 'K', 'k'>(noarr::lit<8>);

	auto ta = orig_ta ^ i_blocks;
	auto tb = orig_tb ^ k_blocks;
	auto tc = orig_tc ^ i_blocks ^ k_blocks;

	auto trav = noarr::traverser(ta, tb, tc).order(noarr::bcast<'1'>(1));
	auto td = noarr::scalar<num_t>() ^ noarr::vectors_like<'k', 'i'>(trav.top_struct());

	auto cutrav = noarr::cuda_threads<'I', 'i', 'K', '1'>(trav);

	cutrav.simple_run(kernel_matmul, td | noarr::get_size(), ta, tb, tc, td, pa, pb, pc);
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

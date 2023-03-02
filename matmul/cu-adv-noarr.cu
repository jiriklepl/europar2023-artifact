#define CUDA
#include "noarrmain.hpp"

template<class T, class TA, class TB, class TC, class TD>
__global__ void matmul(T trav, TA ta, TB tb, TC tc, TD td, num_t *pa, num_t *pb, num_t *pc) {
	extern __shared__ char pd[];

	trav.template for_dims<'j'>([=](auto inner) {
		td | noarr::get_at(pd, inner.state()) = 0;
	});

	trav.template for_dims<'k', 'j'>([=](auto inner) {
		auto state = inner.state();
		td | noarr::get_at(pd, state) += (ta | noarr::get_at(pa, state)) * (tb | noarr::get_at(pb, state));
	});

	trav.template for_dims<'j'>([=](auto inner) {
		auto state = inner.state();
		tc | noarr::get_at(pc, state) = td | noarr::get_at(pd, state);
	});
}

template<class A, class B, class C>
void run_matmul(A orig_ta, B orig_tb, C orig_tc, num_t *pa, num_t *pb, num_t *pc) {
	auto i_blocks = noarr::into_blocks<'i', 'I', 'i'>(noarr::lit<1024>);
	auto j_blocks = noarr::into_blocks<'j', 'J', 'j'>(noarr::lit<8>);

	auto ta = orig_ta ^ i_blocks;
	auto tb = orig_tb ^ j_blocks;
	auto tc = orig_tc ^ i_blocks ^ j_blocks;

	auto trav = noarr::traverser(ta, tb, tc).order(noarr::bcast<'1'>(1));
	auto td = noarr::scalar<num_t>() ^ noarr::vectors_like<'j', 'i'>(trav.top_struct());

	auto cutrav = noarr::cuda_threads<'I', 'i', 'J', '1'>(trav);

	cutrav.simple_run(matmul, td | noarr::get_size(), ta, tb, tc, td, pa, pb, pc);
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

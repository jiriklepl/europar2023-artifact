#define CUDA
#include "matmulmain.hpp"

template<typename T, typename TA, typename TB, typename TC, typename TD>
__global__ void kernel_matmul(T trav, TA ta, TB tb, TC tc, TD td, void *pa, void *pb, void *pc) {
	extern __shared__ char pd[];

	trav.order(noarr::reorder<'k'>()).for_each([=](auto ijk) {
		td | noarr::get_at(pd, ijk) = 0;
	});

	trav.order(noarr::reorder<'j', 'k'>()).for_each([=](auto ijk) {
		num_t a_elem = ta | noarr::get_at(pa, ijk);
		num_t b_elem = tb | noarr::get_at(pb, ijk);
		td | noarr::get_at(pd, ijk) += a_elem * b_elem;
	});

	trav.order(noarr::reorder<'k'>()).for_each([=](auto ijk) {
		num_t c_elem = td | noarr::get_at(pd, ijk);
		tc | noarr::get_at(pc, ijk) = c_elem;
	});
}

template<typename A, typename B, typename C>
void matmul(A orig_ta, B orig_tb, C orig_tc, char *pa, char *pb, char *pc) {
	auto i_blocks = noarr::into_blocks<'i', /*'r',*/ 'I', 'i'>(noarr::lit<1024>);
	auto k_blocks = noarr::into_blocks<'k', /*'s',*/ 'K', 'k'>(noarr::lit<8>);

	auto ta = orig_ta ^ i_blocks;
	auto tb = orig_tb ^ k_blocks;
	auto tc = orig_tc ^ i_blocks ^ k_blocks;

	noarr::traverser(tc).order(noarr::reorder</*'s', 'r'*/>()).for_each([=](auto rs) {
		auto td = noarr::scalar<num_t>()
			^ noarr::sized_vector<'i'>(tc | noarr::get_length<'i'>(rs))
			^ noarr::sized_vector<'k'>(tc | noarr::get_length<'k'>(rs));

		auto trav = noarr::cuda_traverser(ta, tb, tc, td).order(noarr::fix(rs) ^ noarr::bcast<'1'>(1)).template threads<'I', 'i', 'K', '1'>();
		kernel_matmul<<<trav.grid_dim(), trav.block_dim(), td | noarr::get_size()>>>(trav.inner(), ta, tb, tc, td, pa, pb, pc);

		CUCH(cudaGetLastError());
	});

	CUCH(cudaDeviceSynchronize());
}

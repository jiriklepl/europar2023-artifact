#define CUDA
#include "matmulmain.hpp"

template<typename T, typename A, typename B, typename C, typename TD>
__global__ void kernel_matmul(T trav, A a, B b, C c, TD td) {
	extern __shared__ char pd[];
	auto d = noarr::make_bag(td, pd);

	trav.order(noarr::reorder<'k'>()).for_each([&](auto ijk) {
		d[ijk] = 0;
	});

	trav.order(noarr::reorder<'j', 'k'>()).for_each([&](auto ijk) {
		d[ijk] += a[ijk] * b[ijk];
	});

	trav.order(noarr::reorder<'k'>()).for_each([&](auto ijk) {
		c[ijk] = d[ijk];
	});
}

template<typename A, typename B, typename C>
void matmul(A orig_ta, B orig_tb, C orig_tc, char *pa, char *pb, char *pc) {
	auto i_blocks = noarr::into_blocks<'i', /*'r',*/ 'I', 'i'>(noarr::lit<1024>);
	auto k_blocks = noarr::into_blocks<'k', /*'s',*/ 'K', 'k'>(noarr::lit<8>);

	auto a = noarr::make_bag(orig_ta ^ i_blocks, pa);
	auto b = noarr::make_bag(orig_tb ^ k_blocks, pb);
	auto c = noarr::make_bag(orig_tc ^ i_blocks ^ k_blocks, pc);

	noarr::traverser(c).order(noarr::reorder</*'s', 'r'*/>()).for_each([=](auto rs) {
		auto td = noarr::scalar<num_t>()
			^ noarr::sized_vector<'i'>(c.template get_length<'i'>(rs))
			^ noarr::sized_vector<'k'>(c.template get_length<'k'>(rs));

		auto trav = noarr::cuda_traverser(a, b, c, td).order(noarr::fix(rs) ^ noarr::bcast<'1'>(1)).template threads<'I', 'i', 'K', '1'>();
		kernel_matmul<<<trav.grid_dim(), trav.block_dim(), td | noarr::get_size()>>>(trav.inner(), a, b, c, td);

		CUCH(cudaGetLastError());
	});

	CUCH(cudaDeviceSynchronize());
}

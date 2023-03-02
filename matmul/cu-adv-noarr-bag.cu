#define CUDA
#include "noarrmain.hpp"

#include <noarr/structures/interop/bag.hpp>

template<class T, class A, class B, class C, class TD>
__global__ void matmul(T trav, A a, B b, C c, TD td) {
	extern __shared__ char pd[];
	auto d = noarr::make_bag(td, pd);

	trav.template for_dims<'j'>([=](auto inner) {
		d[inner.state()] = 0;
	});

	trav.template for_dims<'k', 'j'>([=](auto inner) {
		auto state = inner.state();
		d[state] += a[state] * b[state];
	});

	trav.template for_dims<'j'>([=](auto inner) {
		auto state = inner.state();
		c[state] = d[state];
	});
}

template<class A, class B, class C>
void run_matmul(A ta, B tb, C tc, num_t *pa, num_t *pb, num_t *pc) {
	auto i_blocks = noarr::into_blocks<'i', 'I', 'i'>(noarr::lit<1024>);
	auto j_blocks = noarr::into_blocks<'j', 'J', 'j'>(noarr::lit<8>);

	auto a = noarr::make_bag(ta ^ i_blocks, pa);
	auto b = noarr::make_bag(tb ^ j_blocks, pb);
	auto c = noarr::make_bag(tc ^ i_blocks ^ j_blocks, pc);

	auto trav = noarr::traverser(a, b, c).order(noarr::bcast<'1'>(1));
	auto td = noarr::scalar<num_t>() ^ noarr::vectors_like<'j', 'i'>(trav.top_struct());

	auto cutrav = noarr::cuda_threads<'I', 'i', 'J', '1'>(trav);

	cutrav.simple_run(matmul, td | noarr::get_size(), a, b, c, td);
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

#define CPU
#include "../matmul/noarrmain.hpp"

template<class A, class B, class C>
void run_matmul(A ta, B tb, C tc, num_t *pa, num_t *pb, num_t *pc) {
	auto a = noarr::make_bag(ta, pa);
	auto b = noarr::make_bag(tb, pb);
	auto c = noarr::make_bag(tc, pc);

	// PAPER: 3.1 First example
	noarr::traverser(c).for_each([=](auto state) { c[state] = 0; });

	// PAPER: 3.1 Third example
	auto into_blocks = noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<16>)
		^ noarr::strip_mine<'k', 'K', 'k'>(noarr::lit<16>)
		^ noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<16>);

	noarr::traverser(a, b, c).order(into_blocks).for_each(
		[=](auto state) { c[state] += a[state] * b[state]; });
}

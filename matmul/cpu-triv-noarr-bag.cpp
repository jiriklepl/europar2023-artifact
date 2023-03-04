#define CPU
#include "noarrmain.hpp"

#include <noarr/structures/interop/bag.hpp>

template<class A, class B, class C>
void run_matmul(A ta, B tb, C tc, num_t *pa, num_t *pb, num_t *pc) {
	auto a = noarr::make_bag(ta, pa);
	auto b = noarr::make_bag(tb, pb);
	auto c = noarr::make_bag(tc, pc);

	LOG("# reset c");
	noarr::traverser(c).for_each([=](auto state) {
		LOG("push 0");
		LOG("store c at i=" << noarr::get_index<'i'>(state) << " j=" << noarr::get_index<'j'>(state));
		c[state] = 0;
	});

	// PAPER: 3.1 Second example (LOGs serve as comments)
	LOG("# multiply a and b, add the result to c");
	noarr::traverser(a, b, c).for_each([=](auto state) {
		LOG("load a at i=" << noarr::get_index<'i'>(state) << " k=" << noarr::get_index<'k'>(state));
		LOG("load b at k=" << noarr::get_index<'k'>(state) << " j=" << noarr::get_index<'j'>(state));
		LOG("multiply");
		LOG("add");
		LOG("store c at i=" << noarr::get_index<'i'>(state) << " j=" << noarr::get_index<'j'>(state));
		c[state] += a[state] * b[state];
	});
}

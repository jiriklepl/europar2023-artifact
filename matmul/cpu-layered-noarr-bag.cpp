#define CPU
#include "noarrmain.hpp"

template<class A, class B, class C>
void matmul(A ta, B tb, C tc, char *pa, char *pb, char *pc) {
	auto a = noarr::make_bag(ta, pa);
	auto b = noarr::make_bag(tb, pb);
	auto c = noarr::make_bag(tc, pc);

	auto into_blocks = noarr::into_blocks<'k', 'K', 'k'>(noarr::lit<16>)
		^ noarr::into_blocks<'j', 'J', 'j'>(noarr::lit<16>);

	noarr::traverser(c).for_each([=](auto state) {
		c[state] = 0;
	});

	noarr::traverser(a, b, c)
		.order(into_blocks)
		.template for_dims<'K', 'J', 'k', 'i'>([=](auto inner) {
		auto result = c[inner.state()];

		inner.for_each([=, &result](auto state){
			result += a[state] * b[state];
		});

		c[inner.state()] = result;
	});
}

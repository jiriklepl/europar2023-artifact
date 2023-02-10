#define CPU
#include "noarrmain.hpp"

template<class A, class B, class C>
void matmul(A ta, B tb, C tc, char *pa, char *pb, char *pc) {
	auto a = noarr::make_bag(ta, pa);
	auto b = noarr::make_bag(tb, pb);
	auto c = noarr::make_bag(tc, pc);

    auto into_blocks = noarr::strip_mine<'k', 'K', 'k'>(noarr::lit<16>)
        ^ noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<16>);

    noarr::traverser(c).for_each([=](auto state) {
	    c[state] = 0;
    });

    noarr::traverser(a, b, c).order(into_blocks).for_each([=](auto state) {
        c[state] += a[state] * b[state];
    });
}

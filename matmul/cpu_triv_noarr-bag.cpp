#define CPU
#include "noarrmain.hpp"

template<class A, class B, class C>
void matmul(A orig_ta, B orig_tb, C orig_tc, char *pa, char *pb, char *pc) {
	auto a = noarr::make_bag(orig_ta, pa);
	auto b = noarr::make_bag(orig_tb, pb);
	auto c = noarr::make_bag(orig_tc, pc);

    noarr::traverser(c).for_each([=](auto state) {
	    c[state] = 0;
    });

    noarr::traverser(a, b, c).for_each([=](auto state) {
        c[state] += a[state] * b[state];
    });
}

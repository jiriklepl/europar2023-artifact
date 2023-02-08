#define CPU
#include "noarrmain.hpp"

template<typename A, typename B, typename C>
void matmul(A orig_ta, B orig_tb, C orig_tc, char *pa, char *pb, char *pc) {
	auto a = noarr::make_bag(orig_ta, pa);
	auto b = noarr::make_bag(orig_tb, pb);
	auto c = noarr::make_bag(orig_tc, pc);

    noarr::traverser(c).for_each([=](auto ik) {
	    c[ik] = 0;
    });

    noarr::traverser(a, b, c).for_each([=](auto ijk) {
        c[ijk] += a[ijk] * b[ijk];
    });
}

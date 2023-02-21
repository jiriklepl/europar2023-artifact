#define CPU
#include "noarrmain.hpp"

template<class A, class B, class C>
void matmul(A ta, B tb, C tc, char *pa, char *pb, char *pc) {
	noarr::traverser(tc).for_each([=](auto state) {
		tc | noarr::get_at(pc, state) = 0;
	});

	noarr::traverser(ta, tb, tc).for_each([=](auto state) {
		num_t a_elem = ta | noarr::get_at(pa, state);
		num_t b_elem = tb | noarr::get_at(pb, state);
		tc | noarr::get_at(pc, state) += a_elem * b_elem;
	});
}

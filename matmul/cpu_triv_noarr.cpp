#define CPU
#include "noarrmain.hpp"

template<typename A, typename B, typename C>
void matmul(A ta, B tb, C tc, char *pa, char *pb, char *pc) {
    noarr::traverser(tc).for_each([=](auto ik) {
	    tc | noarr::get_at(pc, ik) = 0;
    });

	noarr::traverser(ta, tb, tc).for_each([=](auto ijk) {
        num_t a_elem = ta | noarr::get_at(pa, ijk);
        num_t b_elem = tb | noarr::get_at(pb, ijk);
        tc | noarr::get_at(pc, ijk) += a_elem * b_elem;
    });
}

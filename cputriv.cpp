#define CPU
#include "matmulmain.hpp"

template<typename TC>
constexpr auto kernel_reset(TC tc, void *pc) {
    return [=](auto state) {
	    tc | noarr::get_at(pc, state) = 0;
    };
}

template<typename TA, typename TB, typename TC>
constexpr auto kernel_matmul(TA ta, TB tb, TC tc, void *pa, void *pb, void *pc) {
    return [=](auto trav) {
        num_t result = tc | noarr::get_at(pc, trav.state());

        trav.for_each([=, &result](auto ijk) {
            num_t a_elem = ta | noarr::get_at(pa, ijk);
            num_t b_elem = tb | noarr::get_at(pb, ijk);
            result += a_elem * b_elem;
        });

        tc | noarr::get_at(pc, trav.state()) = result;
    };
}

template<typename A, typename B, typename C>
void matmul(A orig_ta, B orig_tb, C orig_tc, char *pa, char *pb, char *pc) {
#ifdef BLOCK_I
	auto i_blocks = noarr::into_blocks<'i', 'I', 'i'>(noarr::lit<16>);
#else
    auto i_blocks = noarr::bcast<'I'>(1);
#endif
#ifdef BLOCK_J
	auto j_blocks = noarr::into_blocks<'j', 'J', 'j'>(noarr::lit<16>);
#else
    auto j_blocks = noarr::bcast<'J'>(1);
#endif
#ifdef BLOCK_K
	auto k_blocks = noarr::into_blocks<'k', 'K', 'k'>(noarr::lit<16>);
#else
    auto k_blocks = noarr::bcast<'K'>(1);
#endif

	auto ta = orig_ta ^ i_blocks ^ j_blocks;
	auto tb = orig_tb ^ j_blocks ^ k_blocks;
	auto tc = orig_tc ^ i_blocks ^ k_blocks;

    noarr::traverser(tc).for_each(kernel_reset(tc, pc));

	auto trav = noarr::traverser(ta, tb, tc);
    trav.template for_dims<'I', 'K', 'J', 'i', 'k'>(kernel_matmul(ta, tb, tc, pa, pb, pc));
}

#define CPU
#include "matmulmain.hpp"

template<typename C>
constexpr auto kernel_reset(C c) {
    return [=](auto state) {
	    c[state] = 0;
    };
}

template<typename A, typename B, typename C>
constexpr auto kernel_matmul(A a, B b, C c) {
    return [=](auto trav) {
        num_t result = c[trav.state()];

        trav.for_each([=, &result](auto ijk) {
            result += a[ijk] * b[ijk];
        });

        c[trav.state()] = result;
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
	auto a = noarr::make_bag(orig_ta ^ i_blocks ^ j_blocks, pa);
	auto b = noarr::make_bag(orig_tb ^ j_blocks ^ k_blocks, pb);
	auto c = noarr::make_bag(orig_tc ^ i_blocks ^ k_blocks, pc);

    noarr::traverser(c).for_each(kernel_reset(c));

	auto trav = noarr::traverser(a, b, c);
    trav.template for_dims<'I', 'K', 'J', 'i', 'k'>(kernel_matmul(a, b, c));
}

#define CPU
#include "noarrmain.hpp"

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

template<class F, std::size_t ...Idxs>
constexpr auto transform_pack(F f, std::index_sequence<Idxs...>) {
    return std::integer_sequence<
        typename decltype(f(std::integral_constant<std::size_t, 0>()))::value_type,
        decltype(f(std::integral_constant<std::size_t, Idxs>()))::value...>();
}

template<std::size_t I, std::size_t J, class C, C ...Idxs>
constexpr auto swap_pack(std::integer_sequence<C, Idxs...>) {
    constexpr std::size_t l = std::min(I, J);
    constexpr std::size_t h = std::max(I, J);
    constexpr C idxs[] = {Idxs...};

    return transform_pack([&]<std::size_t X>(std::integral_constant<std::size_t, X>) {
        if constexpr(X != l && X != h)
            return std::integral_constant<C, idxs[X]>();
        else if constexpr(X == l)
            return std::integral_constant<C, idxs[h]>();
        else
            return std::integral_constant<C, idxs[l]>();
    }, std::make_index_sequence<sizeof...(Idxs)>());
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

#ifndef BLOCK_ORDER
#error BLOCK_ORDER has to satisfy: 0 <= BLOCK_ORDER < 6
#elif BLOCK_ORDER >= 6 or BLOCK_ORDER < 0
#error BLOCK_ORDER has to satisfy: 0 <= BLOCK_ORDER < 6
#endif
#ifndef DIM_ORDER
#error DIM_ORDER has to satisfy: 0 <= DIM_ORDER < 2
#elif DIM_ORDER >= 2 or DIM_ORDER < 0
#error DIM_ORDER has to satisfy: 0 <= DIM_ORDER < 2
#endif
	auto trav = noarr::traverser(a, b, c);
    // trav.template for_dims<'I', J', 'K', 'i', 'k'>(kernel_matmul(a, b, c));
    // modified for the experiment:
    [&]<char ...Blocks, char ...Dims>(std::integer_sequence<char, Blocks...>, std::integer_sequence<char, Dims...>){
        trav.template for_dims<Blocks..., Dims...>(kernel_matmul(a, b, c));
    }(swap_pack<1, 1 + (BLOCK_ORDER / 3)>(swap_pack<0, BLOCK_ORDER % 3>(std::integer_sequence<char, 'I', 'J', 'K'>())), swap_pack<0, DIM_ORDER>(std::integer_sequence<char, 'i', 'k'>()));
}

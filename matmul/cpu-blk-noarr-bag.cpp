#define CPU
#define POLY
#include "noarrmain.hpp"

#include <noarr/structures/interop/bag.hpp>

template<class C>
constexpr auto kernel_reset(C c) {
	return [=](auto state) {
		LOG("push 0");
		LOG("store c at i=" << noarr::get_index<'i'>(state) << " k=" << noarr::get_index<'k'>(state));
		c[state] = 0;
	};
}

template<class A, class B, class C>
constexpr auto kernel_matmul(A a, B b, C c) {
	return [=](auto trav) {
		LOG("load c at i=" << noarr::get_index<'i'>(trav.state()) << " k=" << noarr::get_index<'k'>(trav.state()));
		num_t result = c[trav.state()];

		trav.for_each([=, &result](auto ijk) {
			LOG("load a at i=" << noarr::get_index<'i'>(ijk) << " j=" << noarr::get_index<'j'>(ijk));
			LOG("load b at j=" << noarr::get_index<'j'>(ijk) << " k=" << noarr::get_index<'k'>(ijk));
			LOG("multiply");
			LOG("add");
			result += a[ijk] * b[ijk];
		});

		LOG("store c at i=" << noarr::get_index<'i'>(trav.state()) << " k=" << noarr::get_index<'k'>(trav.state()));
		c[trav.state()] = result;
	};
}

template<class A, class B, class C>
void matmul(A ta, B tb, C tc, char *pa, char *pb, char *pc) {
#ifdef BLOCK_I
	auto i_blocks = noarr::into_blocks<'i', 'I', 'i'>(noarr::lit<BLOCK_SIZE>);
#else
	auto i_blocks = noarr::bcast<'I'>(noarr::lit<1>);
#endif
#ifdef BLOCK_J
	auto j_blocks = noarr::into_blocks<'j', 'J', 'j'>(noarr::lit<BLOCK_SIZE>);
#else
	auto j_blocks = noarr::bcast<'J'>(noarr::lit<1>);
#endif
#ifdef BLOCK_K
	auto k_blocks = noarr::into_blocks<'k', 'K', 'k'>(noarr::lit<BLOCK_SIZE>);
#else
	auto k_blocks = noarr::bcast<'K'>(noarr::lit<1>);
#endif
	auto a = noarr::make_bag(ta, pa);
	auto b = noarr::make_bag(tb, pb);
	auto c = noarr::make_bag(tc, pc);

	LOG("# reset c");
	noarr::traverser(c)
		.for_each(kernel_reset(c));

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
	LOG("# multiply a and b, add the result to c");
	auto trav = noarr::traverser(a, b, c)
		.order(i_blocks ^ j_blocks ^ k_blocks);

	// trav.template for_dims<'I', J', 'K', 'i', 'k'>(kernel_matmul(a, b, c));
	// modified for the experiments:
	[=]<char ...Blocks, char ...Dims>(std::integer_sequence<char, Blocks...>, std::integer_sequence<char, Dims...>){
		trav.template for_dims<Blocks..., Dims...>(kernel_matmul(a, b, c));
	}(swap_pack<1, 1 + (BLOCK_ORDER / 3)>(swap_pack<0, BLOCK_ORDER % 3>(std::integer_sequence<char, 'I', 'J', 'K'>())), swap_pack<0, DIM_ORDER>(std::integer_sequence<char, 'i', 'k'>()));
}

#define CPU
#include "histomain.hpp"

#include <span>
#include <ranges>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/traverser_iter.hpp>
#include <noarr/structures/interop/bag.hpp>

#ifndef HISTO_IMPL
#error Add -DHISTO_IMPL=(histo_loop|histo_range|histo_foreach|histo_tbbreduce) to compiler commandline
#define HISTO_IMPL histo_undefined
#endif

#ifdef HISTO_HAVE_TBB
#include <noarr/structures/interop/tbb.hpp>
#endif

namespace {

enum {
	histo_loop,
	histo_range,
	histo_foreach,
	histo_tbbreduce,
	histo_undefined
};

}

void histo(void *in_ptr, std::size_t size, void *out_ptr) {

if constexpr (HISTO_IMPL == histo_loop) {
	using noarr::idx;

	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), (char *)in_ptr);
	auto out = noarr::make_bag(noarr::scalar<std::size_t>() ^ noarr::array<'v', 256>(), (char *)out_ptr);

	for(std::size_t i = 0; i < size; ++i) {
		value_t value = in[idx<'i'>(i)];
		out[idx<'v'>(value)] += 1;
	}
}

if constexpr (HISTO_IMPL == histo_range) {
	using noarr::idx;

	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), (char *)in_ptr);
	auto out = noarr::make_bag(noarr::scalar<std::size_t>() ^ noarr::array<'v', 256>(), (char *)out_ptr);

	for(auto elem : noarr::traverser(in)) {
		value_t value = in[elem.state()];
		out[idx<'v'>(value)] += 1;
	}
}

if constexpr (HISTO_IMPL == histo_foreach) {
	using noarr::idx;

	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), (char *)in_ptr);
	auto out = noarr::make_bag(noarr::scalar<std::size_t>() ^ noarr::array<'v', 256>(), (char *)out_ptr);

	noarr::traverser(in).for_each([in, out](auto in_state) {
		value_t value = in[in_state];
		out[idx<'v'>(value)] += 1;
	});
}

#ifdef HISTO_HAVE_TBB
if constexpr (HISTO_IMPL == histo_tbbreduce) {
	using noarr::idx;

	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), (char *)in_ptr);
	auto out = noarr::make_bag(noarr::scalar<std::size_t>() ^ noarr::array<'v', 256>(), (char *)out_ptr);

	noarr::tbb_reduce_bag(
		// Input traverser.
		noarr::traverser(in),

		// Neutralizing function, OutElem := 0
		[](auto out_state, auto &out_left) {
			out_left[out_state] = 0;
		},

		// Accumulation function, Out += InElem
		[in](auto in_state, auto &out_left) {
			value_t value = in[in_state];
			out_left[idx<'v'>(value)] += 1;
		},

		// Joining function, OutElem += OutElem
		[](auto out_state, auto &out_left, const auto &out_right) {
			out_left[out_state] += out_right[out_state];
		},

		// Output bag.
		out
	);
}
#endif

}

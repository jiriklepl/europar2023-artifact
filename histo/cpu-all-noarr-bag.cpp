#define CPU
#include "histomain.hpp"

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

enum {
	histo_loop,
	histo_range,
	histo_foreach,
	histo_tbbreduce,
	histo_undefined
};

void run_histogram(value_t *in_ptr, std::size_t size, std::size_t *out_ptr) {

if constexpr (HISTO_IMPL == histo_loop) {
	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), in_ptr);
	auto out = noarr::make_bag(noarr::scalar<std::size_t>() ^ noarr::array<'v', 256>(), out_ptr);

	for(std::size_t i = 0; i < size; ++i) {
		value_t value = in[noarr::idx<'i'>(i)];
		out[noarr::idx<'v'>(value)] += 1;
	}
}

else if constexpr (HISTO_IMPL == histo_range) {
	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), in_ptr);
	auto out = noarr::make_bag(noarr::scalar<std::size_t>() ^ noarr::array<'v', 256>(), out_ptr);

	for(auto elem : noarr::traverser(in)) {
		value_t value = in[elem.state()];
		out[noarr::idx<'v'>(value)] += 1;
	}
}

else if constexpr (HISTO_IMPL == histo_foreach) {
	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), in_ptr);
	auto out = noarr::make_bag(noarr::scalar<std::size_t>() ^ noarr::array<'v', 256>(), out_ptr);

	// PAPER 3.2 First example
	noarr::traverser(in).for_each([in, out](auto in_state) {
		value_t value = in[in_state];
		out[noarr::idx<'v'>(value)] += 1;
	});
}

#ifdef HISTO_HAVE_TBB
else if constexpr (HISTO_IMPL == histo_tbbreduce) {
	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), in_ptr);
	auto out = noarr::make_bag(noarr::scalar<std::size_t>() ^ noarr::array<'v', 256>(), out_ptr);

	// PAPER 3.2 Second example
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
			out_left[noarr::idx<'v'>(value)] += 1;
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

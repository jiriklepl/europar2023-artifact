#define CPU
#include "histomain.hpp"

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/traverser_iter.hpp>
#include <noarr/structures/interop/bag.hpp>

#ifndef HISTO_IMPL
#error Add -DHISTO_IMPL=(histo_plain_loop|histo_noarr_loop|histo_trav_loop|histo_trav_foreach|histo_trav_tbbreduce) to compiler commandline
#define HISTO_IMPL hist_undefined
#endif

#ifdef HISTO_HAVE_TBB
#include <noarr/structures/interop/tbb.hpp>
#endif

namespace {

enum {
	histo_plain_loop,
	histo_noarr_loop,
	histo_trav_loop,
	histo_trav_foreach,
	histo_trav_tbbreduce,
	hist_undefined
};

}

void histo(void *in_ptr, size_t size, void *out_ptr) {

if constexpr (HISTO_IMPL == histo_plain_loop) {
	using noarr::get_at;

	auto in = (value_t*) in_ptr;
	auto out = (size_t*) out_ptr;

	for(size_t i = 0; i < size; i++) {
		value_t value = in[i];
		out[value] += 1;
	}
}

if constexpr (HISTO_IMPL == histo_noarr_loop) {
	using noarr::idx;

	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), (char *)in_ptr);
	auto out = noarr::make_bag(noarr::scalar<size_t>() ^ noarr::array<'v', 256>(), (char *)out_ptr);

	for(size_t i = 0; i < size; i++) {
		value_t value = in[idx<'i'>(i)];
		out[idx<'v'>(value)] += 1;
	}
}

if constexpr (HISTO_IMPL == histo_trav_loop) {
	using noarr::idx;

	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), (char *)in_ptr);
	auto out = noarr::make_bag(noarr::scalar<size_t>() ^ noarr::array<'v', 256>(), (char *)out_ptr);

	for(auto elem : noarr::traverser(in)) {
		value_t value = in[elem.state()];
		out[idx<'v'>(value)] += 1;
	}
}

if constexpr (HISTO_IMPL == histo_trav_foreach) {
	using noarr::idx;

	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), (char *)in_ptr);
	auto out = noarr::make_bag(noarr::scalar<size_t>() ^ noarr::array<'v', 256>(), (char *)out_ptr);

	noarr::traverser(in).for_each([in, out](auto in_state) {
		value_t value = in[in_state];
		out[idx<'v'>(value)] += 1;
	});
}

#ifdef HISTO_HAVE_TBB
if constexpr (HISTO_IMPL == histo_trav_tbbreduce) {
	using noarr::idx;

	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), (char *)in_ptr);
	auto out = noarr::make_bag(noarr::scalar<size_t>() ^ noarr::array<'v', 256>(), (char *)out_ptr);

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

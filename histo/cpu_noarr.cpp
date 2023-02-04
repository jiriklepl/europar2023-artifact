#define CPU
#include "histomain.hpp"

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/traverser_iter.hpp>

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

else if constexpr (HISTO_IMPL == histo_noarr_loop) {
	using noarr::get_at;

	auto in = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out = noarr::scalar<size_t>() ^ noarr::array<'v', 256>();

	for(size_t i = 0; i < size; i++) {
		value_t value = in | get_at<'i'>(in_ptr, i);
		out | get_at<'v'>(out_ptr, value) += 1;
	}
}

else if constexpr (HISTO_IMPL == histo_trav_loop) {
	using noarr::get_at;

	auto in = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out = noarr::scalar<size_t>() ^ noarr::array<'v', 256>();

	for(auto elem : noarr::traverser(in)) {
		value_t value = in | get_at(in_ptr, elem.state());
		out | get_at<'v'>(out_ptr, value) += 1;
	}
}

else if constexpr (HISTO_IMPL == histo_trav_foreach) {
	using noarr::get_at;

	auto in = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out = noarr::scalar<size_t>() ^ noarr::array<'v', 256>();

	noarr::traverser(in).for_each([in, in_ptr, out, out_ptr](auto in_state) {
		value_t value = in | get_at(in_ptr, in_state);
		out | get_at<'v'>(out_ptr, value) += 1;
	});
}

#ifdef HISTO_HAVE_TBB
else if constexpr (HISTO_IMPL == histo_trav_tbbreduce) {
	using noarr::get_at;

	auto in = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out = noarr::scalar<size_t>() ^ noarr::array<'v', 256>();

	noarr::tbb_reduce(
		// Input traverser.
		noarr::traverser(in),

		// Neutralizing function, OutElem := 0
		[out](auto out_state, void *out_left) {
			out | get_at(out_left, out_state) = 0;
		},

		// Accumulation function, Out += InElem
		[in, in_ptr, out](auto in_state, void *out_left) {
			value_t value = in | get_at(in_ptr, in_state);
			out | get_at<'v'>(out_left, value) += 1;
		},

		// Joining function, OutElem += OutElem
		[out](auto out_state, void *out_left, const void *out_right) {
			out | get_at(out_left, out_state) += out | get_at(out_right, out_state);
		},

		// Output type.
		out,

		// Output pointer.
		out_ptr
	);
}
#endif

}

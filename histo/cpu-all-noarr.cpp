#define CPU
#include "histomain.hpp"

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/traverser_iter.hpp>

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
	auto in = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out = noarr::scalar<std::size_t>() ^ noarr::array<'v', NUM_VALUES>();

	for(std::size_t i = 0; i < size; ++i) {
		value_t value = in | noarr::get_at<'i'>(in_ptr, i);
		out | noarr::get_at<'v'>(out_ptr, value) += 1;
	}
}

else if constexpr (HISTO_IMPL == histo_range) {
	auto in = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out = noarr::scalar<std::size_t>() ^ noarr::array<'v', NUM_VALUES>();

	for(auto elem : noarr::traverser(in)) {
		value_t value = in | noarr::get_at(in_ptr, elem.state());
		out | noarr::get_at<'v'>(out_ptr, value) += 1;
	}
}

else if constexpr (HISTO_IMPL == histo_foreach) {
	auto in = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out = noarr::scalar<std::size_t>() ^ noarr::array<'v', NUM_VALUES>();

	noarr::traverser(in).for_each([in, in_ptr, out, out_ptr](auto in_state) {
		value_t value = in | noarr::get_at(in_ptr, in_state);
		out | noarr::get_at<'v'>(out_ptr, value) += 1;
	});
}

#ifdef HISTO_HAVE_TBB
else if constexpr (HISTO_IMPL == histo_tbbreduce) {
	auto in = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out = noarr::scalar<std::size_t>() ^ noarr::array<'v', NUM_VALUES>();

	noarr::tbb_reduce(
		// Input traverser.
		noarr::traverser(in),

		// Neutralizing function, OutElem := 0
		[out](auto out_state, void *out_left) {
			out | noarr::get_at(out_left, out_state) = 0;
		},

		// Accumulation function, Out += InElem
		[in, in_ptr, out](auto in_state, void *out_left) {
			value_t value = in | noarr::get_at(in_ptr, in_state);
			out | noarr::get_at<'v'>(out_left, value) += 1;
		},

		// Joining function, OutElem += OutElem
		[out](auto out_state, void *out_left, const void *out_right) {
			out | noarr::get_at(out_left, out_state) += out | noarr::get_at(out_right, out_state);
		},

		// Output type.
		out,

		// Output pointer.
		out_ptr
	);
}
#endif

}

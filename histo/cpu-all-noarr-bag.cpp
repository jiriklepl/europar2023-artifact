#define CPU
#include "histomain.hpp"

#include <span>
#include <ranges>

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
	histo_plain_tbbreduce,
	hist_undefined
};

}

void histo(void *in_ptr, std::size_t size, void *out_ptr) {

if constexpr (HISTO_IMPL == histo_plain_loop) {
	auto in = (value_t*) in_ptr;
	auto out = (std::size_t*) out_ptr;

	for(std::size_t i = 0; i < size; ++i) {
		value_t value = in[i];
		out[value] += 1;
	}
}

if constexpr (HISTO_IMPL == histo_noarr_loop) {
	using noarr::idx;

	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), (char *)in_ptr);
	auto out = noarr::make_bag(noarr::scalar<std::size_t>() ^ noarr::array<'v', 256>(), (char *)out_ptr);

	for(std::size_t i = 0; i < size; ++i) {
		value_t value = in[idx<'i'>(i)];
		out[idx<'v'>(value)] += 1;
	}
}

if constexpr (HISTO_IMPL == histo_trav_loop) {
	using noarr::idx;

	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), (char *)in_ptr);
	auto out = noarr::make_bag(noarr::scalar<std::size_t>() ^ noarr::array<'v', 256>(), (char *)out_ptr);

	for(auto elem : noarr::traverser(in)) {
		value_t value = in[elem.state()];
		out[idx<'v'>(value)] += 1;
	}
}

if constexpr (HISTO_IMPL == histo_trav_foreach) {
	using noarr::idx;

	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), (char *)in_ptr);
	auto out = noarr::make_bag(noarr::scalar<std::size_t>() ^ noarr::array<'v', 256>(), (char *)out_ptr);

	noarr::traverser(in).for_each([in, out](auto in_state) {
		value_t value = in[in_state];
		out[idx<'v'>(value)] += 1;
	});
}

#ifdef HISTO_HAVE_TBB
if constexpr (HISTO_IMPL == histo_trav_tbbreduce) {
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

else if constexpr (HISTO_IMPL == histo_plain_tbbreduce) {
	tbb::combinable<std::unique_ptr<std::size_t[]>> out_ptrs;
	tbb::parallel_for(
		// Input range.
		tbb::blocked_range<std::size_t>(0, size),

		[in_ptr=(value_t *)in_ptr, &out_ptrs](const auto &sub_range) {
			auto &local = out_ptrs.local();

			if (local == nullptr) {
				local = std::make_unique<std::size_t[]>(NUM_VALUES);

				// Neutralizing, OutElem := 0
				for (auto &value : std::span(local.get(), NUM_VALUES))
					value = 0;
			}

			// Accumulation, Out += InElem
			for (auto value : std::span(in_ptr + sub_range.begin(), in_ptr + sub_range.end())) {
				local[value] += 1;
			}
	});

	// Joining, OutElem += OutElem
	out_ptrs.combine_each([out_ptr=(std::size_t *)out_ptr](const auto &local) {
		for (std::size_t i = 0; i != NUM_VALUES; ++i)
			out_ptr[i] += local[i];
	});
}
#endif

}

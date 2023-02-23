#define CPU
#include "histomain.hpp"

#include <algorithm>
#include <span>

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
	auto in = (value_t*) in_ptr;
	auto out = (std::size_t*) out_ptr;

	for(std::size_t i = 0; i < size; ++i) {
		value_t value = in[i];
		out[value] += 1;
	}
}

if constexpr (HISTO_IMPL == histo_range) {
	auto in = (value_t*) in_ptr;
	auto out = (std::size_t*) out_ptr;

	for(value_t value : std::span(in, in + size)) {
		out[value] += 1;
	}
}

else if constexpr (HISTO_IMPL == histo_foreach) {
	auto in = (value_t*) in_ptr;
	auto out = (std::size_t*) out_ptr;

	std::for_each(in, in + size, [out](value_t value) {
		out[value] += 1;
	});
}

#ifdef HISTO_HAVE_TBB
else if constexpr (HISTO_IMPL == histo_tbbreduce) {
	tbb::combinable<std::unique_ptr<std::size_t[]>> out_ptrs;
	tbb::parallel_for(
		// Input range.
		tbb::blocked_range<std::size_t>(0, size),

		[in_ptr=(value_t *)in_ptr, &out_ptrs](const auto &sub_range) {
			auto &local = out_ptrs.local();

			if (local == nullptr) {
				local = std::make_unique<std::size_t[]>(NUM_VALUES);

				// Neutralizing, OutElem := 0
				std::for_each(local.get(), local.get() + NUM_VALUES, [](std::size_t &value) {
					value = 0;
				});
			}

			// Accumulation, Out += InElem
			std::for_each(in_ptr + sub_range.begin(), in_ptr + sub_range.end(), [&local](value_t value) {
				local[value] += 1;
			});
	});

	// Joining, OutElem += OutElem
	out_ptrs.combine_each([out_ptr=(std::size_t *)out_ptr](const auto &local) {
		// sadly, there is no zip until c++23...
		for (std::size_t i = 0; i < NUM_VALUES; ++i)
			out_ptr[i] += local[i];
	});
}
#endif

}

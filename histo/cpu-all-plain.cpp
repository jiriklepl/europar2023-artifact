#define CPU
#include "histomain.hpp"

#include <algorithm>
#include <execution>
#include <span>

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
	for(std::size_t i = 0; i < size; ++i) {
		value_t value = in_ptr[i];
		out_ptr[value] += 1;
	}
}

if constexpr (HISTO_IMPL == histo_range) {
	for(value_t value : std::span(in_ptr, in_ptr + size)) {
		out_ptr[value] += 1;
	}
}

else if constexpr (HISTO_IMPL == histo_foreach) {
	std::for_each_n(std::execution::unseq, in_ptr, size, [out_ptr](value_t value) {
		out_ptr[value] += 1;
	});
}

#ifdef HISTO_HAVE_TBB
else if constexpr (HISTO_IMPL == histo_tbbreduce) {
	tbb::combinable<std::unique_ptr<std::size_t[]>> out_ptrs;
	tbb::parallel_for(
		// Input range.
		tbb::blocked_range<std::size_t>(0, size),

		[in_ptr, &out_ptrs](const auto &sub_range) {
			auto &local = out_ptrs.local();

			if (local == nullptr) {
				local = std::make_unique<std::size_t[]>(NUM_VALUES);

				// Neutralizing, OutElem := 0
				std::for_each_n(std::execution::unseq, local.get(),  NUM_VALUES, [](std::size_t &value) {
					value = 0;
				});
			}

			// Accumulation, Out += InElem
			std::for_each(std::execution::unseq, in_ptr + sub_range.begin(), in_ptr + sub_range.end(), [local=local.get()](value_t value) {
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

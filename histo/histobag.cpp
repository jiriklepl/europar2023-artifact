#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <chrono>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/traverser_iter.hpp>
#include <noarr/structures/interop/bag.hpp>

#ifdef HISTO_HAVE_TBB
#include <noarr/structures/interop/tbb.hpp>
#endif

static constexpr std::size_t NUM_VALUES = 256;

using value_t = unsigned char;

void histo_plain_loop(char *in_ptr, size_t size, char *out_ptr) {
	using noarr::get_at;

	auto in = (value_t*) in_ptr;
	auto out = (size_t*) out_ptr;

	for(size_t i = 0; i < size; i++) {
		value_t value = in[i];
		out[value] += 1;
	}
}

void histo_noarr_loop(char *in_ptr, size_t size, char *out_ptr) {
	using noarr::idx;

	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), in_ptr);
	auto out = noarr::make_bag(noarr::scalar<size_t>() ^ noarr::array<'v', 256>(), out_ptr);

	for(size_t i = 0; i < size; i++) {
		value_t value = in[idx<'i'>(i)];
		out[idx<'v'>(value)] += 1;
	}
}

void histo_trav_loop(char *in_ptr, size_t size, char *out_ptr) {
	using noarr::idx;

	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), in_ptr);
	auto out = noarr::make_bag(noarr::scalar<size_t>() ^ noarr::array<'v', 256>(), out_ptr);

	for(auto elem : noarr::traverser(in)) {
		value_t value = in[elem.state()];
		out[idx<'v'>(value)] += 1;
	}
}

void histo_trav_foreach(char *in_ptr, size_t size, char *out_ptr) {
	using noarr::idx;

	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), in_ptr);
	auto out = noarr::make_bag(noarr::scalar<size_t>() ^ noarr::array<'v', 256>(), out_ptr);

	noarr::traverser(in).for_each([in, out](auto in_state) {
		value_t value = in[in_state];
		out[idx<'v'>(value)] += 1;
	});
}

#ifdef HISTO_HAVE_TBB
void histo_trav_tbbreduce(char *in_ptr, size_t size, char *out_ptr) {
	using noarr::idx;

	auto in = noarr::make_bag(noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size), in_ptr);
	auto out_layout = noarr::scalar<size_t>() ^ noarr::array<'v', 256>();

	noarr::tbb_reduce(
		// Input traverser.
		noarr::traverser(in),

		// Neutralizing function, OutElem := 0
		[out_layout](auto out_state, void *out_left) {
			auto out = noarr::make_bag(out_layout, (char *)out_left);
			out[out_state] = 0;
		},

		// Accumulation function, Out += InElem
		[in, in_ptr, out_layout](auto in_state, void *out_left) {
			auto out = noarr::make_bag(out_layout, (char *)out_left);
			value_t value = in[in_state];
			out[idx<'v'>(value)] += 1;
		},

		// Joining function, OutElem += OutElem
		[out_layout](auto out_state, void *out_left, const void *out_right) {
			auto left_bag = noarr::make_bag(out_layout, (char *)out_left);
			auto right_bag = noarr::make_bag(out_layout, (char *)out_right);
			left_bag[out_state] += right_bag[out_state];
		},

		// Output type.
		out_layout,

		// Output pointer.
		out_ptr
	);
}
#endif

#ifndef HISTO_IMPL
#error Add -DHISTO_IMPL=(histo_plain_loop|histo_noarr_loop|histo_trav_loop|histo_trav_foreach|histo_trav_tbbreduce) to compiler commandline
#endif

volatile auto histo_ptr = &HISTO_IMPL;

using namespace std::literals::chrono_literals;

int main(int argc, char **argv) {
	if(argc != 3) {
		std::cerr << "Usage: histo <filename> <size>" << std::endl;
		std::abort();
	}
	std::size_t size = std::stoll(argv[2]);

	void *data = std::malloc(size);

	std::FILE *file = std::fopen(argv[1], "r");
	if(std::fread(data, 1, size, file) != size) {
		std::cerr << "Input error" << std::endl;
		std::abort();
	}
	std::fclose(file);

	std::size_t counts[NUM_VALUES] = {0};

	auto t0 = std::chrono::steady_clock::now();
	histo_ptr((char *)data, size, (char *)counts);
	auto t1 = std::chrono::steady_clock::now();
	std::fprintf(stderr, "%lu.%03u ms\n", (unsigned long) ((t1 - t0) / 1ms), (unsigned) ((t1 - t0) / 1us % 1000));

	std::free(data);

	for(std::size_t v = 0; v < NUM_VALUES; v++) {
		std::printf("%12zu * 0x%02x", counts[v], (unsigned) v);
		if(v >= ' ' && v <= '~')
			std::printf(" ('%c')", (char) v);
		std::putchar('\n');
	}

	return 0;
}

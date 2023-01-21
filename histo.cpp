#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <chrono>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/traverser_iter.hpp>

#ifdef HISTO_HAVE_TBB
#include <noarr/structures/interop/tbb.hpp>
#endif

static constexpr std::size_t NUM_VALUES = 256;

using value_t = unsigned char;

void histo_plain_loop(void *in_ptr, size_t size, void *out_ptr) {
	using noarr::get_at;

	auto in = (value_t*) in_ptr;
	auto out = (size_t*) out_ptr;

	for(size_t i = 0; i < size; i++) {
		value_t value = in[i];
		out[value] += 1;
	}
}

void histo_noarr_loop(void *in_ptr, size_t size, void *out_ptr) {
	using noarr::get_at;

	auto in = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out = noarr::scalar<size_t>() ^ noarr::array<'v', 256>();

	for(size_t i = 0; i < size; i++) {
		value_t value = in | get_at<'i'>(in_ptr, i);
		out | get_at<'v'>(out_ptr, value) += 1;
	}
}

void histo_trav_loop(void *in_ptr, size_t size, void *out_ptr) {
	using noarr::get_at;

	auto in = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out = noarr::scalar<size_t>() ^ noarr::array<'v', 256>();

	for(auto elem : noarr::traverser(in)) {
		value_t value = in | get_at(in_ptr, elem.state());
		out | get_at<'v'>(out_ptr, value) += 1;
	}
}

void histo_trav_foreach(void *in_ptr, size_t size, void *out_ptr) {
	using noarr::get_at;

	auto in = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out = noarr::scalar<size_t>() ^ noarr::array<'v', 256>();

	noarr::traverser(in).for_each([in, in_ptr, out, out_ptr](auto in_state) {
		value_t value = in | get_at(in_ptr, in_state);
		out | get_at<'v'>(out_ptr, value) += 1;
	});
}

#ifdef HISTO_HAVE_TBB
void histo_trav_tbbreduce(void *in_ptr, size_t size, void *out_ptr) {
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
	histo_ptr(data, size, counts);
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

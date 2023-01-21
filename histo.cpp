#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <chrono>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/traverser_iter.hpp>

#ifdef HISTO_HAVE_TBB
#include <noarr/structures/interop/tbb.hpp>
#endif

typedef unsigned char u8;

void histo_plain_loop(void *in_ptr, size_t size, void *out_ptr) {
	using noarr::get_at;

	auto in = (u8*) in_ptr;
	auto out = (size_t*) out_ptr;

	for(size_t i = 0; i < size; i++) {
		u8 value = in[i];
		out[value] += 1;
	}
}

void histo_noarr_loop(void *in_ptr, size_t size, void *out_ptr) {
	using noarr::get_at;

	auto in = noarr::scalar<u8>() ^ noarr::sized_vector<'i'>(size);
	auto out = noarr::scalar<size_t>() ^ noarr::array<'v', 256>();

	for(size_t i = 0; i < size; i++) {
		u8 value = in | get_at<'i'>(in_ptr, i);
		out | get_at<'v'>(out_ptr, value) += 1;
	}
}

void histo_trav_loop(void *in_ptr, size_t size, void *out_ptr) {
	using noarr::get_at;

	auto in = noarr::scalar<u8>() ^ noarr::sized_vector<'i'>(size);
	auto out = noarr::scalar<size_t>() ^ noarr::array<'v', 256>();

	for(auto elem : noarr::traverser(in)) {
		u8 value = in | get_at(in_ptr, elem.state());
		out | get_at<'v'>(out_ptr, value) += 1;
	}
}

void histo_trav_foreach(void *in_ptr, size_t size, void *out_ptr) {
	using noarr::get_at;

	auto in = noarr::scalar<u8>() ^ noarr::sized_vector<'i'>(size);
	auto out = noarr::scalar<size_t>() ^ noarr::array<'v', 256>();

	noarr::traverser(in).for_each([in, in_ptr, out, out_ptr](auto in_state) {
		u8 value = in | get_at(in_ptr, in_state);
		out | get_at<'v'>(out_ptr, value) += 1;
	});
}

#ifdef HISTO_HAVE_TBB
void histo_trav_tbbreduce(void *in_ptr, size_t size, void *out_ptr) {
	using noarr::get_at;

	auto in = noarr::scalar<u8>() ^ noarr::sized_vector<'i'>(size);
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
			u8 value = in | get_at(in_ptr, in_state);
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
	FILE *file = fopen(argv[1], "r");
	size_t size = std::atoi(argv[2]);
	void *data = std::malloc(size);
	size_t counts[256] = {0};
	if(fread(data, 1, size, file) != size) {
		std::cerr << "Input error" << std::endl;
		std::abort();
	}
	fclose(file);

	auto t0 = std::chrono::steady_clock::now();
	histo_ptr(data, size, counts);
	auto t1 = std::chrono::steady_clock::now();
	fprintf(stderr, "%lu.%03u ms\n", (unsigned long) ((t1 - t0) / 1ms), (unsigned) ((t1 - t0) / 1us % 1000));

	std::free(data);
	for(size_t v = 0; v < 256; v++) {
		printf("%12zu * 0x%02x", counts[v], (unsigned) v);
		if(v >= ' ' && v <= '~')
			printf(" ('%c')", (char) v);
		putchar('\n');
	}
	return 0;
}

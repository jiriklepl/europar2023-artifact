#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <utility>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/serialize_data.hpp>
#include <noarr/structures/interop/bag.hpp>

typedef float val_t;

template<char ...Dims, class Radius, class ...StateItems>
constexpr auto neighborhood(Radius radius, noarr::state<StateItems...> state) noexcept {
	return (... ^ noarr::slice<Dims>(noarr::get_index<Dims>(state) - radius,  2 * radius + 1));
}

template<class Radius, class InImage, class OutImage>
void run_stencil(Radius radius, InImage in_image, OutImage out_image) {
	noarr::traverser(in_image)
		.order(noarr::symmetric_slice<'h', 'w'>(in_image.structure().unwrap(), radius))
		.for_each([=](auto state) {
			val_t sum = 0;

			auto count = (2 * radius + 1) * (2 * radius + 1);
			auto fix_channel = noarr::fix<'c'>(state);

			// we create a new traverser because we might choose a different order of iteration
			noarr::traverser(in_image)
				.order(fix_channel ^ neighborhood<'w', 'h'>(radius, state))
				.for_each([=, &sum](auto state) { sum += in_image[state]; });

			out_image[state] += sum / count;
	});
}

int main(int argc, char **argv) {
	if(argc != 6) {
		std::cerr << "Usage: " << argv[0] << " <INPUT> <WIDTH> <HEIGHT> <CHANNELS> <RADIUS>" << std::endl;
		std::abort();
	}

	std::size_t width = std::atoi(argv[2]);
	std::size_t height = std::atoi(argv[3]);
	std::size_t channels = std::atoi(argv[4]);
	std::size_t radius = std::atoi(argv[5]);

	auto image_structure = noarr::scalar<val_t>()
		^ noarr::sized_vector<'c'>(channels)
		^ noarr::sized_vector<'w'>(width)
		^ noarr::sized_vector<'h'>(height);

	auto in_image_handle = noarr::make_bag(image_structure);
	auto in_image = in_image_handle.get_ref();

	auto out_image_handle = noarr::make_bag(image_structure);
	auto out_image = out_image_handle.get_ref();

	noarr::traverser(image_structure).for_each([=](auto state) {
		out_image[state] = 0;
	});

	// Load data.
	{
		std::ifstream file(argv[1]);

		if(!deserialize_data(file, in_image)) {
			std::cerr << "Input error" << std::endl;
			std::abort();
		}
	}

	auto start = std::chrono::high_resolution_clock::now();
	run_stencil(radius, in_image, out_image);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

	std::cerr << duration.count() << std::endl;

	if(!serialize_data(std::cout, out_image)) {
		std::cerr << "Output error" << std::endl;
		std::abort();
	}

	return 0;
}

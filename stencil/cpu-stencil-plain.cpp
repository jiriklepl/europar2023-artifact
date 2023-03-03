#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <utility>

typedef float val_t;

template<class Radius, class Height, class Width, class Channels, class InGet, class OutGet>
void run_stencil(Radius radius, Height height, Width width, Channels channels, InGet in_get, val_t *in_image, OutGet out_get, val_t *out_image) {
	for (std::size_t h = radius; h < height - radius; h++)
		for (std::size_t w = radius; w < width - radius; w++)
			for (std::size_t c = 0; c < channels; c++) {
				val_t sum = 0;

				auto count = (2 * radius + 1) * (2 * radius + 1);

				// we create a new traverser because we might choose a different order of iteration
				for (std::size_t nh = h - radius; nh <= h + radius; nh++)
					for (std::size_t nw = w - radius; nw <= w + radius; nw++)
						sum += in_get(in_image, nh, nw, c);

				out_get(out_image, h, w, c) += sum / count;
	}
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

	auto get = [=](val_t *ptr, auto h, auto w, auto c) -> val_t& { return ptr[h * width * channels + w * channels + c]; };

	auto count = height * width * channels;

	auto in_image_handle = std::make_unique<val_t[]>(count);
	auto in_image = in_image_handle.get();

	auto out_image_handle = std::make_unique<val_t[]>(count);
	auto out_image = out_image_handle.get();

	for (std::size_t h = 0; h < height; h++)
		for (std::size_t w = 0; w < width; w++)
			for (std::size_t c = 0; c < channels; c++)
				get(out_image, h, w, c) = 0;

	// Load data.
	{
		std::ifstream file(argv[1]);

		for (std::size_t h = 0; h < height; h++)
			for (std::size_t w = 0; w < width; w++)
				for (std::size_t c = 0; c < channels; c++)
					file >> get(in_image, h, w, c);

		if(!file) {
			std::cerr << "Input error" << std::endl;
			std::abort();
		}
	}

	auto start = std::chrono::high_resolution_clock::now();
	run_stencil(radius, height, width, channels, get, in_image, get, out_image);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

	std::cerr << duration.count() << std::endl;

	for (std::size_t h = 0; h < height; h++)
		for (std::size_t w = 0; w < width; w++)
			for (std::size_t c = 0; c < channels; c++)
				std::cout << get(out_image, h, w, c) << std::endl;

	if(!std::cout) {
		std::cerr << "Output error" << std::endl;
		std::abort();
	}

	return 0;
}

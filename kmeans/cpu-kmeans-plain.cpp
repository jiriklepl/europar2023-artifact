#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>

typedef int num_t;
typedef float val_t;

auto kmeans(num_t n, num_t k, num_t d) {
	auto points = std::make_unique<val_t[]>(d * n);

	// Load data.
	for (num_t p = 0; p < n; p++)
		for (num_t j = 0; j < d; j++)
			std::cin >> points[p * d + j];

	if (!std::cin) {
		std::cerr << "Input error" << std::endl;
		std::abort();
	}

	auto start = std::chrono::high_resolution_clock::now();

	auto centroids = std::make_unique<val_t[]>(d * k);
	auto sums = std::make_unique<val_t[]>(d * k);
	auto counts = std::make_unique<num_t[]>(k);
	auto asgn = std::make_unique<num_t[]>(n);

	// Initialize centroids.
	// Only works if we initialize c-th centroid with c-th point.
	for (num_t c = 0; c < k; c++)
		for (num_t j = 0; j < d; j++) {
			// Points are normally indexed with 'p' - we will temporarily index them with 'c', i.e. the current cluster.
			centroids[c * d + j] = points[c * d + j];
	}

	for (int i = 0; i < 10; i++) {
		// One iteration of the algorithm consists of two steps.

		// Step 1. Recalculate assignments.
		for (num_t p = 0; p < n; p++) {
			// Select the point we currently want to assign.
			val_t *pth_point = points.get() + p * d;

			// Current closest cluster and the squared distance from its centroid.
			val_t min_dist_sq = INFINITY;
			num_t min_asgn = 0;

			// For each cluster c:
			for (num_t c = 0; c < k; c++) {
				// Calculate the squared distance from the cluster centroid.
				val_t dist_sq = 0;

				// It is the sum of squared differences in individual coordinates j.
				for (num_t j = 0; j < d; j++) {
					// This runs for each coordinate separately.
					val_t a = centroids[c * d + j];
					val_t b = pth_point[j];
					val_t diff = a - b;

					dist_sq += diff * diff;
				}

				if (dist_sq < min_dist_sq) {
					min_dist_sq = dist_sq;
					min_asgn = c;
				}
			}

			asgn[p] = min_asgn;
		}

		// Step 2. Recalculate centroids.
		std::for_each_n(sums.get(), d * k, [](val_t &value) {
			value = 0;
		});

		std::for_each_n(counts.get(), k, [](num_t &value) {
			value = 0;
		});

		for (num_t p = 0; p < n; p++) {
			num_t cluster = asgn[p];

			val_t *point = points.get() + p * d;
			val_t *sum = sums.get() + cluster * d;

			for (num_t j = 0; j < d; j++) {
				sum[j] += point[j];
			}

			counts[cluster] += 1;
		}

		for (num_t c = 0; c < k; c++) {
			num_t count = counts[c];

			if (count) {
				for (num_t j = 0; j < d; j++) {
					centroids[c * d + j] = sums[c * d + j] / count;
				}
			}
		}
	}

	auto end = std::chrono::high_resolution_clock::now();

	std::cout.precision(0);

	// Write results.
	std::cout << std::fixed << "== MEANS ==" << std::endl;
	for (num_t c = 0; c < k; c++)
		for (num_t j = 0; j < d; j++)
			std::cout << centroids[c * d + j] << std::endl;

	if (!std::cout) {
		std::cerr << "Output error" << std::endl;
		std::abort();
	}

	std::cout << std::fixed << "== ASSIGNMENTS ==" << std::endl;
	for (num_t p = 0; p < n; p++)
		std::cout << asgn[p] << std::endl;

	if (!std::cout) {
		std::cerr << "Output error" << std::endl;
		std::abort();
	}

	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	return duration;
}

int main(int argc, char **argv) {
	if (argc != 4) {
		std::cerr << "Usage: kmeans <n points> <k clusters> <d dims>" << std::endl;
		std::abort();
	}

	auto duration = kmeans(std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]));

	std::cerr << duration.count() << std::endl;

	return 0;
}

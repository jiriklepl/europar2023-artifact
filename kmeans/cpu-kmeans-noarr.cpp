#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

typedef int num_t;
typedef float val_t;

auto kmeans(num_t n, num_t k, num_t d) {
	using noarr::get_at;

	auto per_point = noarr::sized_vector<'p'>(n);
	auto per_cluster = noarr::sized_vector<'c'>(k);
	auto per_dim = noarr::sized_vector<'j'>(d);

	auto points = noarr::scalar<val_t>() ^ per_dim ^ per_point;
	void *points_ptr = std::malloc(points | noarr::get_size());

	// Load data.
	if(!deserialize_data(std::cin, points, points_ptr)) {
		std::cerr << "Input error" << std::endl;
		std::abort();
	}

	auto start = std::chrono::high_resolution_clock::now();

	auto centroids = noarr::scalar<val_t>() ^ per_dim ^ per_cluster;
	auto sums = noarr::scalar<val_t>() ^ per_dim ^ per_cluster;
	auto counts = noarr::scalar<num_t>() ^ per_cluster;
	auto asgn = noarr::scalar<num_t>() ^ per_point;

	void *centroids_ptr = std::malloc(centroids | noarr::get_size());
	void *sums_ptr = std::malloc(sums | noarr::get_size());
	void *counts_ptr = std::malloc(counts | noarr::get_size());
	void *asgn_ptr = std::malloc(asgn | noarr::get_size());

	// Initialize centroids.
	// Only works if we initialize c-th centroid with c-th point.
	noarr::traverser(centroids).for_each([=](auto state) {
		// Points are normally indexed with 'p' - we will temporarily index them with 'c', i.e. the current cluster.
		auto points_viewed_as_centroids = points ^ noarr::rename<'p', 'c'>();
		centroids | get_at(centroids_ptr, state) = points_viewed_as_centroids | get_at(points_ptr, state);
	});

	for(int i = 0; i < 10; i++) {
		// One iteration of the algorithm consists of two steps.

		// Step 1. Recalculate assignments.
		noarr::traverser(asgn).for_each([=](auto state_p) {
			// Select the point we currently want to assign.
			auto pth_point = points ^ noarr::fix(state_p);

			// Current closest cluster and the squared distance from its centroid.
			val_t min_dist_sq = INFINITY;
			num_t min_asgn = 0;

			// For each cluster c:
			noarr::traverser(centroids).template for_dims<'c'>([=, &min_dist_sq, &min_asgn](auto inner) {
				// Calculate the squared distance from the cluster centroid.
				val_t dist_sq = 0;

				// It is the sum of squared differences in individual coordinates j.
				inner.for_each([=, &dist_sq](auto state_j) {
					// This runs for each coordinate separately.
					val_t a = centroids | get_at(centroids_ptr, state_j);
					val_t b = pth_point | get_at(points_ptr, state_j);
					val_t diff = a - b;

					dist_sq += diff * diff;
				});

				if(dist_sq < min_dist_sq) {
					min_dist_sq = dist_sq;
					min_asgn = noarr::get_index<'c'>(inner.state());
				}
			});

			asgn | get_at(asgn_ptr, state_p) = min_asgn;
		});

		// Step 2. Recalculate centroids.
		noarr::traverser(sums).for_each([=](auto cluster_dim) {
			sums | get_at(sums_ptr, cluster_dim) = 0;
		});

		noarr::traverser(counts).for_each([=](auto cluster) {
			counts | get_at(counts_ptr, cluster) = 0;
		});

		noarr::traverser(asgn).for_each([=](auto pnt) {
			num_t cluster = asgn | get_at(asgn_ptr, pnt);

			auto point = points ^ noarr::fix(pnt);
			auto sum = sums ^ noarr::fix<'c'>(cluster);

			noarr::traverser(sum).for_each([=](auto dim) {
				sum | get_at(sums_ptr, dim) += point | noarr::get_at(points_ptr, dim);
			});

			counts | get_at<'c'>(counts_ptr, cluster) += 1;
		});

		noarr::traverser(centroids).template for_dims<'c'>([=](auto inner) {
			num_t count = counts | get_at(counts_ptr, inner.state());

			if(count) {
				inner.for_each([=](auto cluster_dim) {
					centroids | get_at(centroids_ptr, cluster_dim) = (sums | get_at(sums_ptr, cluster_dim)) / count;
				});
			}
		});
	}

	auto end = std::chrono::high_resolution_clock::now();

	std::cout.precision(0);

	// Write results.
	if(!serialize_data(std::cout << std::fixed << "== MEANS ==" << std::endl, centroids, centroids_ptr)) {
		std::cerr << "Output error" << std::endl;
		std::abort();
	}

	if(!serialize_data(std::cout << std::fixed << "== ASSIGNMENTS ==" << std::endl, asgn, asgn_ptr)) {
		std::cerr << "Output error" << std::endl;
		std::abort();
	}

	std::free(points_ptr);
	std::free(centroids_ptr);
	std::free(sums_ptr);
	std::free(counts_ptr);
	std::free(asgn_ptr);

	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	return duration;
}

int main(int argc, char **argv) {
	if(argc != 4) {
		std::cerr << "Usage: kmeans <n points> <k clusters> <d dims>" << std::endl;
		std::abort();
	}

	auto duration = kmeans(std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]));

	std::cerr << duration.count() << std::endl;

	return 0;
}

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/serialize_data.hpp>
#include <noarr/structures/interop/bag.hpp>

typedef int num_t;
typedef float val_t;

auto kmeans(num_t n, num_t k, num_t d) {
	auto per_point = noarr::sized_vector<'p'>(n);
	auto per_cluster = noarr::sized_vector<'c'>(k);
	auto per_dim = noarr::sized_vector<'j'>(d);

	auto points_owned = noarr::make_bag(noarr::scalar<val_t>() ^ per_dim ^ per_point);
	auto points = points_owned.get_ref();

	// Load data.
	if(!deserialize_data(std::cin, points)) {
		std::cerr << "Input error" << std::endl;
		std::abort();
	}

	auto start = std::chrono::high_resolution_clock::now();

	auto centroids_owned = noarr::make_bag(noarr::scalar<val_t>() ^ per_dim ^ per_cluster);
	auto sums_owned = noarr::make_bag(noarr::scalar<val_t>() ^ per_dim ^ per_cluster);
	auto counts_owned = noarr::make_bag(noarr::scalar<num_t>() ^ per_cluster);
	auto asgn_owned = noarr::make_bag(noarr::scalar<num_t>() ^ per_point);

	auto centroids = centroids_owned.get_ref();
	auto sums = sums_owned.get_ref();
	auto counts = counts_owned.get_ref();
	auto asgn = asgn_owned.get_ref();

	// Initialize centroids.
	// Only works if we initialize c-th centroid with c-th point.
	noarr::traverser(centroids).for_each([=](auto state) {
		// Points are normally indexed with 'p' - we will temporarily index them with 'c', i.e. the current cluster.
		auto points_viewed_as_centroids = points ^ noarr::rename<'p', 'c'>();
		centroids[state] = points_viewed_as_centroids[state];
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
					val_t a = centroids[state_j];
					val_t b = pth_point[state_j];
					val_t diff = a - b;

					dist_sq += diff * diff;
				});

				if(dist_sq < min_dist_sq) {
					min_dist_sq = dist_sq;
					min_asgn = noarr::get_index<'c'>(inner.state());
				}
			});

			asgn[state_p] = min_asgn;
		});

		// Step 2. Recalculate centroids.
		noarr::traverser(sums).for_each([=](auto cluster_dim) {
			sums[cluster_dim] = 0;
		});

		noarr::traverser(counts).for_each([=](auto cluster) {
			counts[cluster] = 0;
		});

		noarr::traverser(asgn).for_each([=](auto pnt) {
			auto cluster = noarr::idx<'c'>(asgn[pnt]);

			auto point = points ^ noarr::fix(pnt);
			auto sum = sums ^ noarr::fix(cluster);

			noarr::traverser(sum).for_each([=](auto dim) {
				sum[dim] += point[dim];
			});

			counts[cluster] += 1;
		});

		noarr::traverser(centroids).template for_dims<'c'>([=](auto inner) {
			num_t count = counts[inner.state()];

			if(count) {
				inner.for_each([=](auto cluster_dim) {
					centroids[cluster_dim] = sums[cluster_dim] / count;
				});
			}
		});
	}

	auto end = std::chrono::high_resolution_clock::now();

	std::cout.precision(0);

	// Write results.
	if(!serialize_data(std::cout << std::fixed << "== MEANS ==" << std::endl, centroids)) {
		std::cerr << "Output error" << std::endl;
		std::abort();
	}

	if(!serialize_data(std::cout << std::fixed << "== ASSIGNMENTS ==" << std::endl, asgn)) {
		std::cerr << "Output error" << std::endl;
		std::abort();
	}

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

#include <cstdlib>
#include <cmath>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

typedef int num_t;
typedef float val_t;

void kmeans(num_t n, num_t k, num_t d) {
	using noarr::get_at;

	auto per_point = noarr::sized_vector<'p'>(n);
	auto per_cluster = noarr::sized_vector<'c'>(k);
	auto per_dim = noarr::sized_vector<'j'>(d);

	auto points = noarr::scalar<val_t>() ^ per_dim ^ per_point;
	auto centroids = noarr::scalar<val_t>() ^ per_dim ^ per_cluster;
	auto sums = noarr::scalar<val_t>() ^ per_dim ^ per_cluster;
	auto counts = noarr::scalar<num_t>() ^ per_cluster;
	auto asgn = noarr::scalar<num_t>() ^ per_point;

	void *points_ptr = std::malloc(points | noarr::get_size());
	void *centroids_ptr = std::malloc(centroids | noarr::get_size());
	void *sums_ptr = std::malloc(sums | noarr::get_size());
	void *counts_ptr = std::malloc(counts | noarr::get_size());
	void *asgn_ptr = std::malloc(asgn | noarr::get_size());

	// Load data.
	if(!deserialize_data(std::cin, points, points_ptr)) {
		std::cerr << "Input error" << std::endl;
		std::abort();
	}

	// Initialize centroids.

	// Option 1. Only works if we initialize c-th centroid with c-th point.
	noarr::traverser(centroids).for_each([=](auto state) {
		// Points are normally indexed with 'p' - we will temporarily index them with 'c', i.e. the current cluster.
		auto points_viewed_as_centroids = points ^ noarr::rename<'p', 'c'>();
		centroids | get_at(centroids_ptr, state) = points_viewed_as_centroids | get_at(points_ptr, state);
	});
	// Option 2a.
	noarr::traverser(centroids).for_each([=](auto state) {
		auto [cluster_idx, dim_idx] = noarr::get_indices<'c', 'j'>(state);
		auto point_idx = cluster_idx; // more elaborate mapping possible here (note: recalculated for each dimension, extract if necessary)
		centroids | get_at(centroids_ptr, state) = points | get_at<'p', 'j'>(points_ptr, point_idx, dim_idx);
	});
	// Option 2b.
	noarr::traverser(centroids).for_each([=](auto state) {
		auto [cluster_idx, dim_idx] = noarr::get_indices<'c', 'j'>(state);
		auto point_idx = cluster_idx; // more elaborate mapping possible here (note: recalculated for each dimension, extract if necessary)
		centroids | get_at<'c', 'j'>(centroids_ptr, cluster_idx, dim_idx) = points | get_at<'p', 'j'>(points_ptr, point_idx, dim_idx);
	});
	// Option 3.
	noarr::traverser(noarr::scalar<void>() ^ per_cluster).for_each([=](auto cluster) {
		auto centroid = centroids ^ noarr::fix(cluster);
		auto point = points ^ noarr::fix<'p'>(noarr::get_index<'c'>(cluster)); // more elaborate mapping possible here

		noarr::traverser(centroid).for_each([=](auto dim) {
			centroid | get_at(centroids_ptr, dim) = point | get_at(points_ptr, dim);
		});
	});
	// Option 4.
	for(num_t c = 0; c < k; c++) {
		auto centroid = centroids ^ noarr::fix<'c'>(c);
		auto point = points ^ noarr::fix<'p'>(c); // more elaborate mapping possible here

		noarr::traverser(centroid).for_each([=](auto dim) {
			centroid | get_at(centroids_ptr, dim) = point | get_at(points_ptr, dim);
		});
	}

	for(int i = 0; i < 10; i++) {
		// One iteration of the algorithm consists of two steps.

		// Step 1. Recalculate assignments.
		noarr::traverser(asgn).for_each([=](auto state_p) {
			// Select the point we currently want to assign.
			auto pth_point = points ^ noarr::fix(state_p);

			// Current closest cluster and the squared distance from its centroid.
			val_t min_dist_sq = INFINITY;
			num_t min_asgn;

			// For each cluster c:
			for(num_t c = 0; c < k; c++) {
				// Select the centroid from the cluster.
				auto cth_centroid = centroids ^ noarr::fix<'c'>(c);

				// Calculate the squared distance from the cluster centroid.
				val_t dist_sq = 0;
				// It is the sum of squared differences in individual coordinates j.
				noarr::traverser(cth_centroid).for_each([=, &dist_sq](auto state_j) {
					// This runs for each coordinate separately.
					val_t a = cth_centroid | get_at(centroids_ptr, state_j);
					val_t b = pth_point | get_at(points_ptr, state_j);
					val_t diff = a - b;
					dist_sq += diff * diff;
				});

				if(dist_sq < min_dist_sq) {
					min_dist_sq = dist_sq;
					min_asgn = c;
				}
			}

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
		noarr::traverser(counts).for_each([=](auto cluster) {
			num_t count = counts | get_at(counts_ptr, cluster);
			if(count) {
				noarr::traverser(centroids).order(noarr::fix(cluster)).for_each([=](auto cluster_dim) {
					centroids | get_at(centroids_ptr, cluster_dim) = (sums | get_at(sums_ptr, cluster_dim)) / count;
				});
			}
		});
	}

	// Write results.
	if(!serialize_data(std::cout << "== MEANS ==" << std::endl, centroids, centroids_ptr)) {
		std::cerr << "Output error" << std::endl;
		std::abort();
	}
	if(!serialize_data(std::cout << "== ASSIGNMENTS ==" << std::endl, asgn, asgn_ptr)) {
		std::cerr << "Output error" << std::endl;
		std::abort();
	}

	std::free(points_ptr);
	std::free(centroids_ptr);
	std::free(sums_ptr);
	std::free(counts_ptr);
	std::free(asgn_ptr);
}

int main(int argc, char **argv) {
	if(argc != 4) {
		std::cerr << "Usage: kmeans <n points> <k clusters> <d dims>" << std::endl;
		std::abort();
	}
	kmeans(std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]));
	return 0;
}

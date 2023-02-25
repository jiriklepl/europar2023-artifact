#define CUDA
#include "histomain.hpp"

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/cuda_traverser.cuh>

template<class InTrav, class In, class Out>
__global__ void kernel_histo(InTrav in_trav, In in, Out out) {
	in_trav.for_each([&](auto state) {
		auto value = in[state];
		atomicAdd(&out[noarr::idx<'v'>(value)], 1);
	});
}

void histo(void *in_ptr, std::size_t size, void *out_ptr) {
	auto in_struct = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out_struct = noarr::scalar<std::size_t>() ^ noarr::array<'v', NUM_VALUES>();

	auto in_blk_struct = in_struct
		^ noarr::into_blocks<'i', 'y', 'z'>(noarr::lit<BLOCK_SIZE>)
		^ noarr::into_blocks<'y', 'x', 'y'>(noarr::lit<ELEMS_PER_THREAD>);

	auto in = noarr::make_bag(in_blk_struct, (char *)in_ptr);
	auto out = noarr::make_bag(out_struct, (char *)out_ptr);

	auto ct = noarr::cuda_threads<'x', 'z'>(noarr::traverser(in));

	kernel_histo<<<ct.grid_dim(), ct.block_dim()>>>(ct.inner(), in, out);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

#define CUDA
#include "histomain.hpp"

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/cuda_traverser.cuh>

template<class InTrav, class InStruct, class OutStruct>
__global__ void kernel_histo(InTrav in_trav, InStruct in_struct, OutStruct out_struct, void *in_ptr, void *out_ptr) {
	in_trav.for_each([=](auto state) {
		auto value = in_struct | noarr::get_at(in_ptr, state);
		auto &bin = out_struct | noarr::get_at<'v'>(out_ptr, value);
		atomicAdd(&bin, 1);
	});
}

void histo(void *in_ptr, std::size_t size, void *out_ptr) {
	auto in_struct = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out_struct = noarr::scalar<std::size_t>() ^ noarr::array<'v', NUM_VALUES>();

	auto in_blk = in_struct
		^ noarr::into_blocks<'i', 'y', 'z'>(noarr::lit<BLOCK_SIZE>)
		^ noarr::into_blocks<'y', 'x', 'y'>(noarr::lit<ELEMS_PER_THREAD>);

	auto ct = noarr::cuda_threads<'x', 'z'>(noarr::traverser(in_blk));

	kernel_histo<<<ct.grid_dim(), ct.block_dim()>>>(ct.inner(), in_blk, out_struct, in_ptr, out_ptr);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

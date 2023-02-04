#include "cuhistomain.hpp"

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/cuda_traverser.cuh>
#include <noarr/structures/interop/cuda_striped.cuh>
#include <noarr/structures/interop/cuda_step.cuh>

template<typename InTrav, typename In, typename ShmLayout, typename Out>
__global__ void kernel_histo(InTrav in_trav, In in, ShmLayout shm_layout, Out out) {
	extern __shared__ char shm_ptr[];
	auto shm_bag = make_bag(shm_layout, shm_ptr);

	// A private copy will usually be shared by multiple threads (whenever NUM_COPIES < blockDim.x).
	// For some actions, we would like each memory location to be assigned to only one thread.
	// Let us split each copy further into "subsets", where each subset is owned by exactly one thread.
	// Note that `shm_bag` uses `threadIdx%NUM_COPIES` as the index of copy.
	// We can use the remaining bits, `threadIdx/NUM_COPIES`, as the index of subset within copy.
	std::size_t my_copy_idx = shm_layout.current_stripe_index();
	auto subset = noarr::cuda_step(shm_layout.current_stripe_cg());

	// Zero out shared memory. In this particular case, the access pattern happens
	// to be the same as with the `for(i = threadIdx; i < ...; i += blockDim)` idiom.
	noarr::traverser(shm_bag).order(subset).for_each([&](auto state) {
		shm_bag[state] = 0;
	});

	__syncthreads();

	// Count the elements into the histogram copies in shared memory.
	in_trav.for_each([&](auto state) {
		auto value = in[state];
		atomicAdd(&shm_bag[noarr::idx<'v'>(value)], 1);
	});

	__syncthreads();

	// Reduce the bins in shared memory into global memory.
	noarr::traverser(out).order(noarr::cuda_step_block()).for_each([=](auto state) {
		std::size_t collected = 0;

		for(std::size_t i = 0; i < NUM_COPIES; i++) {
			auto shm_state = state.template with<noarr::cuda_stripe_index>((i + my_copy_idx) % NUM_COPIES);
			collected += shm_bag[shm_state];
		}

		atomicAdd(&out[state], collected);
	});
}

void histo_cuda(void *in_ptr, std::size_t size, void *out_ptr) {
	auto in_layout = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out_layout = noarr::scalar<std::size_t>() ^ noarr::array<'v', NUM_VALUES>();

	auto in_blk_layout = in_layout ^ noarr::into_blocks_static<'i', 'C', 'y', 'z'>(noarr::lit<BLOCK_SIZE>) ^ noarr::into_blocks_static<'y', 'D', 'x', 'y'>(noarr::lit<ELEMS_PER_THREAD>);
	auto shm_layout = out_layout ^ noarr::cuda_striped<NUM_COPIES>();

	auto in = noarr::make_bag(in_blk_layout, (char *)in_ptr);
	auto out = noarr::make_bag(out_layout, (char *)out_ptr);

	noarr::traverser(in).template for_dims<'C', 'D'>([=](auto cd){
		auto ct = noarr::cuda_threads<'x', 'z'>(cd);
#ifdef NOARR_CUDA_HISTO_DEBUG
		std::cerr
			<< (noarr::get_index<'C'>(cd) ? "border" : "body")
			<< " of "
			<< (noarr::get_index<'D'>(cd) ? "border" : "body")
			<< ": len<x> = gridDim = "  << (in_blk ^ noarr::fix(cd) | noarr::get_length<'x'>())
			<< ", len<y> = loopLen = "  << (in_blk ^ noarr::fix(cd) | noarr::get_length<'y'>())
			<< ", len<z> = blockDim = " << (in_blk ^ noarr::fix(cd) | noarr::get_length<'z'>())
			<< std::endl;
		std::cerr << (ct?"if(true)\t":"if(false)\t") << "kernel_histo<<<" << ct.grid_dim().x << ", " << ct.block_dim().x << ", " << (out_striped|noarr::get_size()) << ">>>(...);" <<  << std::endl;
#endif
		if(!ct) return;
		kernel_histo<<<ct.grid_dim(), ct.block_dim(),shm_layout|noarr::get_size()>>>(ct.inner(), in, shm_layout, out);
		CUCH(cudaGetLastError());
#ifdef NOARR_CUDA_HISTO_DEBUG
		CUCH(cudaDeviceSynchronize());
#endif
	});

	CUCH(cudaDeviceSynchronize());
}
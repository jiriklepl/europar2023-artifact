#define CUDA
#include "../histo/histomain.hpp"

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/cuda_traverser.cuh>
#include <noarr/structures/interop/cuda_striped.cuh>
#include <noarr/structures/interop/cuda_step.cuh>

// PAPER: 4.2 First example
template<class InT, class In, class ShmS, class Out>
__global__ void histogram(InT in_trav, In in, ShmS shm_s, Out out) {
	extern __shared__ char shm_ptr[];
	auto shm_bag = make_bag(shm_s, shm_ptr);

	// PAPER: 4.2 Third example
	// A private copy will usually be shared by multiple threads (whenever NUM_COPIES < blockDim.x).
	// For some actions, we would like each memory location to be assigned to only one thread.
	// Let us split each copy further into "subsets", where each subset is owned by exactly one thread.
	// Note that `shm_bag` uses `threadIdx%NUM_COPIES` as the index of copy.
	// We can use the remaining bits, `threadIdx/NUM_COPIES`, as the index of subset within copy.
	std::size_t my_copy_idx = shm_s.current_stripe_index();
	auto subset = noarr::cuda_step(shm_s.current_stripe_cg());

	// Zero out shared memory. In this particular case, the access pattern happens
	// to be the same as with the `for(i = threadIdx; i < ...; i += blockDim)` idiom.
	noarr::traverser(shm_bag).order(subset).for_each([=](auto state) {
		shm_bag[state] = 0;
	});

	__syncthreads();

	// Count the elements into the histogram copies in shared memory.
	in_trav.for_each([=](auto state) {
		auto value = in[state];
		atomicAdd(&shm_bag[noarr::idx<'v'>(value)], 1);
	});

	__syncthreads();

	// PAPER: 4.2 Fourth example
	// Reduce the bins in shared memory into global memory.
	noarr::traverser(out)
		.order(noarr::cuda_step_block())
		.for_each([=](auto state) {
			std::size_t sum = 0;

			for(std::size_t i = 0; i < shm_s.num_stripes(); i++) {
				auto shm_state = state.template with<noarr::cuda_stripe_index>((i + my_copy_idx) % shm_s.num_stripes());
				sum += shm_bag[shm_state];
			}

			atomicAdd(&out[state], sum);
	});
}

void run_histogram(value_t *in_ptr, std::size_t size, std::size_t *out_ptr) {
	auto in_struct = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out_struct = noarr::scalar<std::size_t>() ^ noarr::array<'v', NUM_VALUES>();

	// PAPER: 4.1 Second example
	auto in_blk_struct = in_struct
		^ noarr::into_blocks<'i', 'B', 't'>(noarr::lit<BLOCK_SIZE>)
		^ noarr::into_blocks<'B', 'b', 'x'>(noarr::lit<ELEMS_PER_THREAD>);
	auto in = noarr::make_bag(in_blk_struct, in_ptr);
	auto out = noarr::make_bag(out_struct, out_ptr);

	// PAPER: 4.2 Second example
	auto ct = noarr::cuda_threads<'b', 't'>(noarr::traverser(in));
	auto shm_s = out_struct ^ noarr::cuda_striped<NUM_COPIES>();

	// Shorter version of
	//   histogram<<<ct.grid_dim(), ct.block_dim(), shm_s | noarr::get_size()>>>(ct.inner(), in, shm_s, out);
	ct.simple_run(histogram, shm_s | noarr::get_size(), in, shm_s, out);
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

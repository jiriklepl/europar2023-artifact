#include "cuhistomain.hpp"

template<typename InTrav, typename InStruct, typename ShmStruct, typename OutStruct>
__global__ void kernel_histo(InTrav in_trav, InStruct in_struct, ShmStruct shm_struct, OutStruct out_struct, void *in_ptr, void *out_ptr) {
	extern __shared__ char shm_ptr[];

	// A private copy will usually be shared by multiple threads (whenever NUM_COPIES < blockDim.x).
	// For some actions, we would like each memory location to be assigned to only one thread.
	// Let us split each copy further into "subsets", where each subset is owned by exactly one thread.
	std::size_t my_copy_idx = shm_struct.current_stripe_index();
	auto subset = noarr::cuda_step(shm_struct.current_stripe_cg());

	// Zero out shared memory. In this particular case, the access pattern happens
	// to be the same as with the `for(i = threadIdx; i < ...; i += blockDim)` idiom.
	noarr::traverser(shm_struct).order(subset).for_each([=](auto state) {
		shm_struct | noarr::get_at(shm_ptr, state) = 0;
	});

	__syncthreads();

	// Count the elements into the histogram copies in shared memory.
	in_trav.for_each([=](auto state) {
		auto value = in_struct | noarr::get_at(in_ptr, state);
		auto &bin = shm_struct | noarr::get_at<'v'>(shm_ptr, value);
		atomicAdd(&bin, 1);
	});

	__syncthreads();

	// Reduce the bins in shared memory into global memory.
	noarr::traverser(out_struct).order(noarr::cuda_step_block()).for_each([=](auto state) {
		std::size_t collected = 0;

		for(std::size_t i = 0; i < NUM_COPIES; i++) {
			auto shm_state = state.template with<noarr::cuda_stripe_index>((i + my_copy_idx) % NUM_COPIES);
			collected += shm_struct | noarr::get_at(shm_ptr, shm_state);
		}

		auto &bin = out_struct | noarr::get_at(out_ptr, state);
		atomicAdd(&bin, collected);
	});
}

void histo_cuda(void *in_ptr, std::size_t size, void *out_ptr) {
	auto in = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out = noarr::scalar<std::size_t>() ^ noarr::array<'v', NUM_VALUES>();

	auto in_blk = in ^ noarr::into_blocks_static<'i', 'C', 'y', 'z'>(BLOCK_SIZE) ^ noarr::into_blocks_static<'y', 'D', 'x', 'y'>(ELEMS_PER_THREAD);
	auto out_striped = out ^ noarr::cuda_striped<NUM_COPIES>();

	noarr::traverser(in_blk).order(noarr::reorder<'C', 'D'>()).for_each([=](auto cd){
		auto ct = noarr::cuda_traverser(in_blk).order(noarr::fix(cd)).template threads<'x', 'z'>();
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
		kernel_histo<<<ct.grid_dim(), ct.block_dim(),out_striped|noarr::get_size()>>>(ct.inner(), in_blk, out_striped, out, in_ptr, out_ptr);
		CUCH(cudaGetLastError());
#ifdef NOARR_CUDA_HISTO_DEBUG
		CUCH(cudaDeviceSynchronize());
#endif
	});

	CUCH(cudaDeviceSynchronize());
}

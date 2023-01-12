#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <cstring>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/cuda.hpp>
#include <noarr/structures/interop/cuda_striped.hpp>

#define CUCH(status)  do { cudaError_t err = status; if (err != cudaSuccess) std::cerr << __FILE__ ":" << __LINE__ << ": error: " << cudaGetErrorString(err) << "\n\t" #status << std::endl, exit(err); } while (false)

static constexpr std::size_t NUM_VALUES = 256;

static constexpr std::size_t BLOCK_SIZE = 64;
static constexpr std::size_t ELEMS_PER_THREAD = 1024;
static constexpr std::size_t ELEMS_PER_BLOCK = BLOCK_SIZE * ELEMS_PER_THREAD;

static constexpr std::size_t NUM_COPIES = 4;

using value_t = unsigned char;

// TODO fix this somehow
namespace {
	// For some reason, "unsigned long long" is different from "unsigned long", although both have the same size.
	// Only "unsigned long long" is supported by `atomicAdd`.
	// Unfortunately, "size_t" is defined as "unsigned long", and is thus not supported.
	__device__ std::size_t atomicAdd(std::size_t *ptr, std::size_t val) {
		// This pointer cast is probably UB.
		::atomicAdd((unsigned long long *)ptr, val);
	}
}

template<typename InTrav, typename InStruct, typename ShmStruct, typename OutStruct>
__global__ void kernel_histo(InTrav in_trav, InStruct in_struct, ShmStruct shm_struct, OutStruct out_struct, void *in_ptr, void *out_ptr) {
	extern __shared__ char shm_ptr[];

	// A private copy will usually be shared by multiple threads (whenever NUM_COPIES < blockDim.x).
	// For some actions, we would like each memory location to be assigned to only one thread.
	// Let us split each copy further into "subsets", where each subset is owned by exactly one thread.
	// Note that `shm_struct` uses `threadIdx%NUM_COPIES` as the index of copy.
	// We can use the remaining bits, `threadIdx/NUM_COPIES`, as the index of subset within copy.
	std::size_t my_copy_idx = threadIdx.x % NUM_COPIES;
	std::size_t num_threads_in_my_copy = (blockDim.x + NUM_COPIES - my_copy_idx - 1) / NUM_COPIES;
	std::size_t my_idx_in_my_copy = threadIdx.x / NUM_COPIES;
	auto subset = noarr::step<'v'>(my_idx_in_my_copy, num_threads_in_my_copy);

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

	// Reduce all copies to the first one.
	for(std::size_t dist = 1; dist < NUM_COPIES; dist *= 2) {
		std::size_t other_copy_idx = my_copy_idx + dist;
		if(other_copy_idx < NUM_COPIES && other_copy_idx < blockDim.x) {
			noarr::traverser(shm_struct).order(subset).for_each([=](auto state) {
				auto &to = shm_struct | noarr::get_at(shm_ptr, state); // my_copy_idx is used implicitly
				auto &from = shm_struct | noarr::get_at(shm_ptr, state.template with<noarr::cuda_stripe_index>(other_copy_idx));
				to += from;
			});
		}
		__syncthreads();
	}

	// Like `subset`, but now *all* threads will access the first copy
	// (*not only* the threads whose `threadIdx.x % NUM_COPIES == 0`).
	// Thus there will be less work for each, and each will make longer strides.
	auto copy0_subset = noarr::step<'v'>(threadIdx.x, blockDim.x);

	// Write the first copy to the global memory.
	noarr::traverser(shm_struct).order(copy0_subset).for_each([=](auto state) {
		auto &from = shm_struct | noarr::get_at(shm_ptr, state.template with<noarr::cuda_stripe_index>(0));
		auto &to = out_struct | noarr::get_at(out_ptr, state);
		atomicAdd(&to, from);
	});
}

void histo_cuda(void *in_ptr, std::size_t size, void *out_ptr) {
	auto in = noarr::scalar<value_t>() ^ noarr::sized_vector<'i'>(size);
	auto out = noarr::scalar<std::size_t>() ^ noarr::array<'v', NUM_VALUES>();

	//auto in_blk = in ^ noarr::into_blocks<'i', 'C', 'x', 'y'>(ELEMS_PER_BLOCK) ^ noarr::into_blocks<'y', 'D', 'y', 'z'>(BLOCK_SIZE);
	auto in_blk = in ^ noarr::into_blocks<'i', 'C', 'y', 'z'>(BLOCK_SIZE) ^ noarr::into_blocks<'y', 'D', 'x', 'y'>(ELEMS_PER_THREAD);
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

volatile auto histo_ptr = &histo_cuda;

using namespace std::literals::chrono_literals;

int main(int argc, char **argv) {
	if(argc != 3) {
		std::cerr << "Usage: histo <filename> <size>" << std::endl;
		std::abort();
	}
	std::size_t size = std::stoll(argv[2]);

	//void *data = std::malloc(size);
	void *data;
	CUCH(cudaMallocManaged(&data, size));

	std::FILE *file = std::fopen(argv[1], "r");
	if(std::fread(data, 1, size, file) != size) {
		std::cerr << "Input error" << std::endl;
		std::abort();
	}
	std::fclose(file);

	//std::size_t counts[NUM_VALUES] = {0};
	std::size_t *counts;
	CUCH(cudaMallocManaged(&counts, NUM_VALUES * sizeof(std::size_t)));
	std::memset(counts, 0, NUM_VALUES * sizeof(std::size_t));

	auto t0 = std::chrono::steady_clock::now();
	histo_ptr(data, size, counts);
	auto t1 = std::chrono::steady_clock::now();
	std::fprintf(stderr, "%lu.%03u ms\n", (unsigned long) ((t1 - t0) / 1ms), (unsigned) ((t1 - t0) / 1us % 1000));

	//std::free(data);
	CUCH(cudaFree(data));

	for(std::size_t v = 0; v < NUM_VALUES; v++) {
		std::printf("%12zu * 0x%02x", counts[v], (unsigned) v);
		if(v >= ' ' && v <= '~')
			std::printf(" ('%c')", (char) v);
		std::putchar('\n');
	}

	CUCH(cudaFree(counts));
	return 0;
}

#define CPU
#define POLY
#include "policymain.hpp"

template<class C>
constexpr auto reset(C c) {
	return [=](auto i, auto j) {
		LOG("push 0");
		LOG("store c at i=" << i << " j=" << j);
		c(j, i) = 0;
	};
}

template<class KStep, class A, class B, class C>
constexpr auto matmul(KStep k_step, A a, B b, C c) {
	return [=](auto i, auto j, auto K) {
		LOG("load c at i=" << i << " j=" << j);
		num_t result = c(j, i);

		for (std::size_t k = 0; k < k_step; k++) {
			LOG("load a at i=" << i << " k=" << K + k);
			LOG("load b at k=" << K + k << " j=" << j);
			LOG("multiply");
			LOG("add");
			result += a(K + k, i) * b(j, K + k);
		}

		LOG("store c at i=" << i << " j=" << j);
		c(j, i) = result;
	};
}

template<class ISize, class JSize, class KSize, class A, class B, class C>
void run_matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c) {
#ifdef BLOCK_I
#define I_LOOP for(std::size_t I = 0; I < i_size / BLOCK_SIZE; I++)
#define I_STEP ((std::size_t)BLOCK_SIZE)
#else
#define I_LOOP if constexpr(const std::size_t I = 0; true)
#define I_STEP i_size
#endif

#ifdef BLOCK_J
#define J_LOOP for(std::size_t J = 0; J < j_size / BLOCK_SIZE; J++)
#define J_STEP ((std::size_t)BLOCK_SIZE)
#else
#define J_LOOP if constexpr(const std::size_t J = 0; true)
#define J_STEP j_size
#endif

#ifdef BLOCK_K
#define K_LOOP for(std::size_t K = 0; K < k_size / BLOCK_SIZE; K++)
#define K_STEP ((std::size_t)BLOCK_SIZE)
#else
#define K_LOOP if constexpr(const std::size_t K = 0; true)
#define K_STEP k_size
#endif
	auto init = reset(c);
	auto body = matmul(K_STEP, a, b, c);

	LOG("# reset c");
	for(std::size_t j = 0; j < j_size; j++)
			for(std::size_t i = 0; i < i_size; i++)
				init(i, j);

	LOG("# multiply a and b, add the result to c");
#ifndef BLOCK_ORDER
#error BLOCK_ORDER has to satisfy: 0 <= BLOCK_ORDER < 6
#elif BLOCK_ORDER == 0
	I_LOOP J_LOOP K_LOOP
#elif BLOCK_ORDER == 1
	J_LOOP I_LOOP K_LOOP
#elif BLOCK_ORDER == 2
	K_LOOP J_LOOP I_LOOP
#elif BLOCK_ORDER == 3
	I_LOOP K_LOOP J_LOOP
#elif BLOCK_ORDER == 4
	J_LOOP K_LOOP I_LOOP
#elif BLOCK_ORDER == 5
	K_LOOP I_LOOP J_LOOP
#else
#error BLOCK_ORDER has to satisfy: 0 <= BLOCK_ORDER < 6
#endif

#ifndef DIM_ORDER
#error DIM_ORDER has to satisfy: 0 <= DIM_ORDER < 2
#elif DIM_ORDER == 0
		for(std::size_t i = 0; i < I_STEP; i++)
			for(std::size_t j = 0; j < J_STEP; j++)
#elif DIM_ORDER == 1
		for(std::size_t j = 0; j < J_STEP; j++)
			for(std::size_t i = 0; i < I_STEP; i++)
#else
#error DIM_ORDER has to satisfy: 0 <= DIM_ORDER < 2
#endif

				body(I * I_STEP + i, J * J_STEP + j, K * K_STEP);
}

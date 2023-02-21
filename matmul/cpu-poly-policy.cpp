#define CPU
#define POLY
#include "policymain.hpp"

template<class C>
constexpr auto kernel_reset(C c) {
	return [=](auto i, auto k) {
		c(k, i) = 0;
	};
}

template<class JStep, class A, class B, class C>
constexpr auto kernel_matmul(JStep j_step, A a, B b, C c) {
	return [=](auto i, auto J, auto k) {
		num_t result = c(k, i);

		for (std::size_t j = 0; j < j_step; j++) {
			result += a(J + j, i) * b(k, J + j);
		}

		c(k, i) = result;
	};
}

template<class ISize, class JSize, class KSize, class A, class B, class C>
void matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c) {
#ifdef BLOCK_I
#define I_LOOP for(std::size_t I = 0; I < i_size / std::integral_constant<std::size_t, BLOCK_SIZE>(); I++)
#define I_STEP std::integral_constant<std::size_t, BLOCK_SIZE>()
#else
#define I_LOOP if constexpr(const std::size_t I = 0; true)
#define I_STEP i_size
#endif

#ifdef BLOCK_J
#define J_LOOP for(std::size_t J = 0; J < j_size / std::integral_constant<std::size_t, BLOCK_SIZE>(); J++)
#define J_STEP std::integral_constant<std::size_t, BLOCK_SIZE>()
#else
#define J_LOOP if constexpr(const std::size_t J = 0; true)
#define J_STEP j_size
#endif

#ifdef BLOCK_K
#define K_LOOP for(std::size_t K = 0; K < k_size / std::integral_constant<std::size_t, BLOCK_SIZE>(); K++)
#define K_STEP std::integral_constant<std::size_t, BLOCK_SIZE>()
#else
#define K_LOOP if constexpr(const std::size_t K = 0; true)
#define K_STEP k_size
#endif
	auto reset = kernel_reset(c);
	auto body = kernel_matmul(J_STEP, a, b, c);

	I_LOOP
		for(std::size_t i = 0; i < I_STEP; i++)
			K_LOOP
				for(std::size_t k = 0; k < K_STEP; k++)
					reset(I * I_STEP + i, K * K_STEP + k);


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
			for(std::size_t k = 0; k < K_STEP; k++)
#elif DIM_ORDER == 1
		for(std::size_t k = 0; k < K_STEP; k++)
			for(std::size_t i = 0; i < I_STEP; i++)
#else
#error DIM_ORDER has to satisfy: 0 <= DIM_ORDER < 2
#endif

				body(K * K_STEP + k, J * J_STEP, I * I_STEP + i);
}

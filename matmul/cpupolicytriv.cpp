#define CPU
#include "policymain.hpp"

#ifdef BLOCK_I
#define I_LOOP for(std::size_t I = 0; I < i_size / I_BLOCK_SIZE; I++)
#define I_STEP I_BLOCK_SIZE
#else
#define I_LOOP for(std::size_t I = 0; I < i_size; I++)
#define I_STEP 1
#endif

#ifdef BLOCK_J
#define J_LOOP for(std::size_t J = 0; J < j_size / J_BLOCK_SIZE; J++)
#define J_STEP J_BLOCK_SIZE
#else
#define J_LOOP for(std::size_t J = 0; J < j_size; J++)
#define J_STEP 1
#endif

#ifdef BLOCK_K
#define K_LOOP for(std::size_t K = 0; K < k_size / K_BLOCK_SIZE; K++)
#define K_STEP K_BLOCK_SIZE
#else
#define K_LOOP for(std::size_t K = 0; K < k_size; K++)
#define K_STEP 1
#endif

template<class C>
constexpr auto kernel_reset(C c) {
    return [=](auto i, auto k) {
        c(k, i) = 0;
    };
}

template<std::size_t JStep, class A, class B, class C>
constexpr auto kernel_matmul(A a, B b, C c) {
    return[=](auto i, auto J, auto k) {
        num_t result = c(k, i);

        for (std::size_t j = J; j < J + JStep; j++) {
            result += a(j, i) * b(k, j);
        }

        c(k, i) = result;
    };
}

template<class ISize, class JSize, class KSize, class A, class B, class C>
void matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c) {
	static constexpr std::size_t I_BLOCK_SIZE = 16;
	static constexpr std::size_t J_BLOCK_SIZE = 16;
	static constexpr std::size_t K_BLOCK_SIZE = 16;
    auto reset = kernel_reset(c);
    auto body = kernel_matmul<J_STEP>(a, b, c);

    for(std::size_t i = 0; i < i_size; i++)
        for(std::size_t k = 0; k < k_size; k++)
            reset(i, k);


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
#error DIM_ORDER has to satisfy: 0 <= DIM_ORDER < 2
#endif

                body(K * K_STEP + k, J * J_STEP, I * I_STEP + i);
}

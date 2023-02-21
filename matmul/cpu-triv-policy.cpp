#define CPU
#include "policymain.hpp"

template<class ISize, class JSize, class KSize, class A, class B, class C>
void matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c) {
	for(std::size_t i = 0; i < i_size; i++)
		for(std::size_t k = 0; k < k_size; k++)
			c(k, i) = 0;

	for(std::size_t k = 0; k < k_size; k++)
		for(std::size_t j = 0; j < j_size; j++)
			for(std::size_t i = 0; i < i_size; i++)
				c(k, i) += a(j, i) * b(k, j);
}

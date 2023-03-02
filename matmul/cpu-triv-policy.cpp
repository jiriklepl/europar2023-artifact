#define CPU
#include "policymain.hpp"

template<class ISize, class JSize, class KSize, class A, class B, class C>
void run_matmul(ISize i_size, JSize j_size, KSize k_size, A a, B b, C c) {
	LOG("# reset c");
	for(std::size_t j = 0; j < j_size; j++)
		for(std::size_t i = 0; i < i_size; i++) {
			LOG("push 0");
			LOG("store c at i=" << i << " j=" << j);
			c(j, i) = 0;
	}

	LOG("# multiply a and b, add the result to c");
	for(std::size_t j = 0; j < j_size; j++)
		for(std::size_t k = 0; k < k_size; k++)
			for(std::size_t i = 0; i < i_size; i++) {
				LOG("load a at i=" << i << " k=" << k);
				LOG("load b at k=" << k << " j=" << j);
				LOG("multiply");
				LOG("add");
				LOG("store c at i=" << i << " j=" << j);
				c(j, i) += a(k, i) * b(j, k);
	}
}

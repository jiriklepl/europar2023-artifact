#define CPU
#include "noarrmain.hpp"

template<class A, class B, class C>
void run_matmul(A ta, B tb, C tc, num_t *pa, num_t *pb, num_t *pc) {
	LOG("# reset c");
	noarr::traverser(tc).for_each([=](auto state) {
		LOG("push 0");
		LOG("store c at i=" << noarr::get_index<'i'>(state) << " j=" << noarr::get_index<'j'>(state));
		tc | noarr::get_at(pc, state) = 0;
	});

	LOG("# multiply a and b, add the result to c");
	noarr::traverser(ta, tb, tc).for_each([=](auto state) {
		LOG("load a at i=" << noarr::get_index<'i'>(state) << " k=" << noarr::get_index<'k'>(state));
		LOG("load b at k=" << noarr::get_index<'k'>(state) << " j=" << noarr::get_index<'j'>(state));
		num_t a_elem = ta | noarr::get_at(pa, state);
		num_t b_elem = tb | noarr::get_at(pb, state);
		LOG("multiply");
		LOG("add");
		LOG("store c at i=" << noarr::get_index<'i'>(state) << " j=" << noarr::get_index<'j'>(state));
		tc | noarr::get_at(pc, state) += a_elem * b_elem;
	});
}

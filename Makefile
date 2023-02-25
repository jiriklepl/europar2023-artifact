GCC := g++
CLANG := clang++
NVCC := nvcc

BUILD_DIR := build

INCLUDE_OPTION := -I noarr-structures/include

CXX_OPTIONS := ${INCLUDE_OPTION} -std=c++20 -Ofast -flto -Wall -Wextra -pedantic -DNDEBUG -march=native -mtune=native
CUDA_OPTIONS := ${INCLUDE_OPTION} -std=c++17 -O3 -dlto --compiler-options -Ofast,-march=native,-mtune=native -DNDEBUG --use_fast_math --expt-relaxed-constexpr

.PHONY: all clean matmul histo kmeans generate generate-small

all: matmul histo kmeans

noarr-structures:
	[ -d noarr-structures ] || git clone https://github.com/ParaCoToUl/noarr-structures.git noarr-structures

matmul: \
	${BUILD_DIR}/matmul/g++/cpu-triv/noarr \
	${BUILD_DIR}/matmul/g++/cpu-triv/noarr-bag \
	${BUILD_DIR}/matmul/g++/cpu-triv/policy \
	${BUILD_DIR}/matmul/clang++/cpu-triv/noarr \
	${BUILD_DIR}/matmul/clang++/cpu-triv/noarr-bag \
	${BUILD_DIR}/matmul/clang++/cpu-triv/policy \
	${BUILD_DIR}/matmul/g++/cpu-blk_ik/noarr \
	${BUILD_DIR}/matmul/g++/cpu-blk_ik/noarr-bag \
	${BUILD_DIR}/matmul/g++/cpu-blk_ik/policy \
	${BUILD_DIR}/matmul/clang++/cpu-blk_ik/noarr \
	${BUILD_DIR}/matmul/clang++/cpu-blk_ik/noarr-bag \
	${BUILD_DIR}/matmul/clang++/cpu-blk_ik/policy \
	${BUILD_DIR}/matmul/g++/cpu-blk_ki/noarr \
	${BUILD_DIR}/matmul/g++/cpu-blk_ki/noarr-bag \
	${BUILD_DIR}/matmul/g++/cpu-blk_ki/policy \
	${BUILD_DIR}/matmul/clang++/cpu-blk_ki/noarr \
	${BUILD_DIR}/matmul/clang++/cpu-blk_ki/noarr-bag \
	${BUILD_DIR}/matmul/clang++/cpu-blk_ki/policy \
	${BUILD_DIR}/matmul/nvcc/cu-basic/noarr \
	${BUILD_DIR}/matmul/nvcc/cu-basic/noarr-bag \
	${BUILD_DIR}/matmul/nvcc/cu-basic/policy \
	${BUILD_DIR}/matmul/nvcc/cu-basic/plain \
	${BUILD_DIR}/matmul/nvcc/cu-adv/noarr \
	${BUILD_DIR}/matmul/nvcc/cu-adv/noarr-bag \
	${BUILD_DIR}/matmul/nvcc/cu-adv/policy \
	${BUILD_DIR}/matmul/nvcc/cu-adv/plain

${BUILD_DIR}/matmul/g++/cpu-triv/%: matmul/cpu-triv-%.cpp noarr-structures matmul/noarrmain.hpp matmul/policymain.hpp
	@mkdir -p $(@D)
	${GCC} -o $@ ${CXX_OPTIONS} $< -DA_ROW -DB_ROW -DC_ROW

${BUILD_DIR}/matmul/clang++/cpu-triv/%: matmul/cpu-triv-%.cpp noarr-structures matmul/noarrmain.hpp matmul/policymain.hpp
	@mkdir -p $(@D)
	${CLANG} -o $@ ${CXX_OPTIONS} $< -DA_ROW -DB_ROW -DC_ROW

${BUILD_DIR}/matmul/g++/cpu-blk_ik/%: matmul/cpu-blk-%.cpp noarr-structures matmul/noarrmain.hpp matmul/policymain.hpp
	@mkdir -p $(@D)
	${GCC} -o $@ ${CXX_OPTIONS} $< -DA_ROW -DB_ROW -DC_ROW -DBLOCK_I -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=0 -DDIM_ORDER=0

${BUILD_DIR}/matmul/clang++/cpu-blk_ik/%: matmul/cpu-blk-%.cpp noarr-structures matmul/noarrmain.hpp matmul/policymain.hpp
	@mkdir -p $(@D)
	${CLANG} -o $@ ${CXX_OPTIONS} $< -DA_ROW -DB_ROW -DC_ROW -DBLOCK_I -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=0 -DDIM_ORDER=0

${BUILD_DIR}/matmul/g++/cpu-blk_ki/%: matmul/cpu-blk-%.cpp noarr-structures matmul/noarrmain.hpp matmul/policymain.hpp
	@mkdir -p $(@D)
	${GCC} -o $@ ${CXX_OPTIONS} $< -DA_ROW -DB_ROW -DC_ROW -DBLOCK_I -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=0 -DDIM_ORDER=1

${BUILD_DIR}/matmul/clang++/cpu-blk_ki/%: matmul/cpu-blk-%.cpp noarr-structures matmul/noarrmain.hpp matmul/policymain.hpp
	@mkdir -p $(@D)
	${CLANG} -o $@ ${CXX_OPTIONS} $< -DA_ROW -DB_ROW -DC_ROW -DBLOCK_I -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=0 -DDIM_ORDER=1

${BUILD_DIR}/matmul/nvcc/cu-basic/%: matmul/cu-basic-%.cu noarr-structures matmul/noarrmain.hpp matmul/policymain.hpp
	@mkdir -p $(@D)
	${NVCC} -o $@ ${CUDA_OPTIONS} $< -DA_ROW -DB_ROW -DC_ROW -DBLOCK_SIZE=32

${BUILD_DIR}/matmul/nvcc/cu-adv/%: matmul/cu-adv-%.cu noarr-structures matmul/noarrmain.hpp matmul/policymain.hpp
	@mkdir -p $(@D)
	${NVCC} -o $@ ${CUDA_OPTIONS} $< -DA_ROW -DB_ROW -DC_ROW

histo: \
	${BUILD_DIR}/histo/g++/cpu-loop/noarr \
	${BUILD_DIR}/histo/g++/cpu-loop/noarr-bag \
	${BUILD_DIR}/histo/g++/cpu-loop/plain \
	${BUILD_DIR}/histo/clang++/cpu-loop/noarr \
	${BUILD_DIR}/histo/clang++/cpu-loop/noarr-bag \
	${BUILD_DIR}/histo/clang++/cpu-loop/plain \
	${BUILD_DIR}/histo/g++/cpu-range/noarr \
	${BUILD_DIR}/histo/g++/cpu-range/noarr-bag \
	${BUILD_DIR}/histo/g++/cpu-range/plain \
	${BUILD_DIR}/histo/clang++/cpu-range/noarr \
	${BUILD_DIR}/histo/clang++/cpu-range/noarr-bag \
	${BUILD_DIR}/histo/clang++/cpu-range/plain \
	${BUILD_DIR}/histo/g++/cpu-foreach/noarr \
	${BUILD_DIR}/histo/g++/cpu-foreach/noarr-bag \
	${BUILD_DIR}/histo/g++/cpu-foreach/plain \
	${BUILD_DIR}/histo/clang++/cpu-foreach/noarr \
	${BUILD_DIR}/histo/clang++/cpu-foreach/noarr-bag \
	${BUILD_DIR}/histo/clang++/cpu-foreach/plain \
	${BUILD_DIR}/histo/g++/cpu-tbb/noarr \
	${BUILD_DIR}/histo/g++/cpu-tbb/noarr-bag \
	${BUILD_DIR}/histo/g++/cpu-tbb/plain \
	${BUILD_DIR}/histo/nvcc/cu-priv/noarr \
	${BUILD_DIR}/histo/nvcc/cu-priv/noarr-bag \
	${BUILD_DIR}/histo/nvcc/cu-priv/plain \
	${BUILD_DIR}/histo/nvcc/cu-triv/noarr \
	${BUILD_DIR}/histo/nvcc/cu-triv/noarr-bag \
	${BUILD_DIR}/histo/nvcc/cu-triv/plain


${BUILD_DIR}/histo/g++/cpu-loop/%: histo/cpu-all-%.cpp noarr-structures histo/histomain.hpp
	@mkdir -p $(@D)
	${GCC} -o $@ ${CXX_OPTIONS} $< -DHISTO_IMPL=histo_loop

${BUILD_DIR}/histo/clang++/cpu-loop/%: histo/cpu-all-%.cpp noarr-structures histo/histomain.hpp
	@mkdir -p $(@D)
	${CLANG} -o $@ ${CXX_OPTIONS} $< -DHISTO_IMPL=histo_loop

${BUILD_DIR}/histo/g++/cpu-range/%: histo/cpu-all-%.cpp noarr-structures histo/histomain.hpp
	@mkdir -p $(@D)
	${GCC} -o $@ ${CXX_OPTIONS} $< -DHISTO_IMPL=histo_range

${BUILD_DIR}/histo/clang++/cpu-range/%: histo/cpu-all-%.cpp noarr-structures histo/histomain.hpp
	@mkdir -p $(@D)
	${CLANG} -o $@ ${CXX_OPTIONS} $< -DHISTO_IMPL=histo_range

${BUILD_DIR}/histo/g++/cpu-foreach/%: histo/cpu-all-%.cpp noarr-structures histo/histomain.hpp
	@mkdir -p $(@D)
	${GCC} -o $@ ${CXX_OPTIONS} $< -DHISTO_IMPL=histo_foreach

${BUILD_DIR}/histo/clang++/cpu-foreach/%: histo/cpu-all-%.cpp noarr-structures histo/histomain.hpp
	@mkdir -p $(@D)
	${CLANG} -o $@ ${CXX_OPTIONS} $< -DHISTO_IMPL=histo_foreach

${BUILD_DIR}/histo/g++/cpu-tbb/%: histo/cpu-all-%.cpp noarr-structures histo/histomain.hpp
	@mkdir -p $(@D)
	${GCC} -o $@ ${CXX_OPTIONS} $< -DHISTO_HAVE_TBB -DHISTO_IMPL=histo_tbbreduce -ltbb

${BUILD_DIR}/histo/nvcc/cu-priv/%: histo/cu-priv-%.cu noarr-structures histo/histomain.hpp
	@mkdir -p $(@D)
	${NVCC} -o $@ ${CUDA_OPTIONS} $<

${BUILD_DIR}/histo/nvcc/cu-triv/%: histo/cu-triv-%.cu noarr-structures histo/histomain.hpp
	@mkdir -p $(@D)
	${NVCC} -o $@ ${CUDA_OPTIONS} $<

${BUILD_DIR}/matmul/matrices_64: matmul/gen-matrices.py
	@mkdir -p $(@D)
	matmul/gen-matrices.py ${BUILD_DIR}/matmul/matrices_64 64

${BUILD_DIR}/matmul/matrices_128: matmul/gen-matrices.py
	@mkdir -p $(@D)
	matmul/gen-matrices.py ${BUILD_DIR}/matmul/matrices_128 128

${BUILD_DIR}/matmul/matrices_256: matmul/gen-matrices.py
	@mkdir -p $(@D)
	matmul/gen-matrices.py ${BUILD_DIR}/matmul/matrices_256 256

${BUILD_DIR}/matmul/matrices_512: matmul/gen-matrices.py
	@mkdir -p $(@D)
	matmul/gen-matrices.py ${BUILD_DIR}/matmul/matrices_512 512

${BUILD_DIR}/matmul/matrices_1024: matmul/gen-matrices.py
	@mkdir -p $(@D)
	matmul/gen-matrices.py ${BUILD_DIR}/matmul/matrices_1024 1024

${BUILD_DIR}/matmul/matrices_2048: matmul/gen-matrices.py
	@mkdir -p $(@D)
	matmul/gen-matrices.py ${BUILD_DIR}/matmul/matrices_2048 2048

${BUILD_DIR}/matmul/matrices_4096: matmul/gen-matrices.py
	@mkdir -p $(@D)
	matmul/gen-matrices.py ${BUILD_DIR}/matmul/matrices_4096 4096

${BUILD_DIR}/matmul/matrices_8192: matmul/gen-matrices.py
	@mkdir -p $(@D)
	matmul/gen-matrices.py ${BUILD_DIR}/matmul/matrices_8192 8192

${BUILD_DIR}/histo/text: histo/gen-text.sh
	@mkdir -p $(@D)
	histo/gen-text.sh ${BUILD_DIR}/histo/text 2G

generate: ${BUILD_DIR}/matmul/matrices_1024 \
	${BUILD_DIR}/matmul/matrices_2048 \
	${BUILD_DIR}/matmul/matrices_4096 \
	${BUILD_DIR}/matmul/matrices_8192 \
	${BUILD_DIR}/histo/text

generate-small: ${BUILD_DIR}/matmul/matrices_64 \
	${BUILD_DIR}/matmul/matrices_128 \
	${BUILD_DIR}/matmul/matrices_256 \
	${BUILD_DIR}/matmul/matrices_512 \
	${BUILD_DIR}/matmul/matrices_1024 \

kmeans: noarr-structures

clean:
	rm -rf ${BUILD_DIR}
	rm -rf noarr-structures

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

matmul: noarr-structures \
	${BUILD_DIR}/matmul/g++/cpu-triv-noarr \
	${BUILD_DIR}/matmul/g++/cpu-triv-noarr-bag \
	${BUILD_DIR}/matmul/g++/cpu-triv-policy \
	${BUILD_DIR}/matmul/clang++/cpu-triv-noarr \
	${BUILD_DIR}/matmul/clang++/cpu-triv-noarr-bag \
	${BUILD_DIR}/matmul/clang++/cpu-triv-policy \
	${BUILD_DIR}/matmul/g++/cpu-poly-noarr \
	${BUILD_DIR}/matmul/g++/cpu-poly-noarr-bag \
	${BUILD_DIR}/matmul/g++/cpu-poly-policy \
	${BUILD_DIR}/matmul/clang++/cpu-poly-noarr \
	${BUILD_DIR}/matmul/clang++/cpu-poly-noarr-bag \
	${BUILD_DIR}/matmul/clang++/cpu-poly-policy \
	${BUILD_DIR}/matmul/g++/cpu-poly-noarr2 \
	${BUILD_DIR}/matmul/g++/cpu-poly-noarr-bag2 \
	${BUILD_DIR}/matmul/g++/cpu-poly-policy2 \
	${BUILD_DIR}/matmul/clang++/cpu-poly-noarr2 \
	${BUILD_DIR}/matmul/clang++/cpu-poly-noarr-bag2 \
	${BUILD_DIR}/matmul/clang++/cpu-poly-policy2 \
	${BUILD_DIR}/matmul/nvcc/cu-basic-noarr \
	${BUILD_DIR}/matmul/nvcc/cu-basic-noarr-bag \
	${BUILD_DIR}/matmul/nvcc/cu-basic-policy \
	${BUILD_DIR}/matmul/nvcc/cu-adv-noarr \
	${BUILD_DIR}/matmul/nvcc/cu-adv-noarr-bag \
	${BUILD_DIR}/matmul/nvcc/cu-adv-policy \
	${BUILD_DIR}/matmul/nvcc/cu-adv-plain

${BUILD_DIR}/matmul/g++/cpu-triv-noarr: matmul/cpu-triv-noarr.cpp matmul/noarrmain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/g++
	${GCC} -o ${BUILD_DIR}/matmul/g++/cpu-triv-noarr ${CXX_OPTIONS} matmul/cpu-triv-noarr.cpp -DA_ROW -DB_ROW -DC_ROW
${BUILD_DIR}/matmul/g++/cpu-triv-noarr-bag: matmul/cpu-triv-noarr-bag.cpp matmul/noarrmain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/g++
	${GCC} -o ${BUILD_DIR}/matmul/g++/cpu-triv-noarr-bag ${CXX_OPTIONS} matmul/cpu-triv-noarr-bag.cpp -DA_ROW -DB_ROW -DC_ROW
${BUILD_DIR}/matmul/g++/cpu-triv-policy: matmul/cpu-triv-policy.cpp matmul/policymain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/g++
	${GCC} -o ${BUILD_DIR}/matmul/g++/cpu-triv-policy ${CXX_OPTIONS} matmul/cpu-triv-policy.cpp -DA_ROW -DB_ROW -DC_ROW

${BUILD_DIR}/matmul/clang++/cpu-triv-noarr: matmul/cpu-triv-noarr.cpp matmul/noarrmain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/clang++
	${CLANG} -o ${BUILD_DIR}/matmul/clang++/cpu-triv-noarr ${CXX_OPTIONS} matmul/cpu-triv-noarr.cpp -DA_ROW -DB_ROW -DC_ROW
${BUILD_DIR}/matmul/clang++/cpu-triv-noarr-bag: matmul/cpu-triv-noarr-bag.cpp matmul/noarrmain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/clang++
	${CLANG} -o ${BUILD_DIR}/matmul/clang++/cpu-triv-noarr-bag ${CXX_OPTIONS} matmul/cpu-triv-noarr-bag.cpp -DA_ROW -DB_ROW -DC_ROW
${BUILD_DIR}/matmul/clang++/cpu-triv-policy: matmul/cpu-triv-policy.cpp matmul/policymain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/clang++
	${CLANG} -o ${BUILD_DIR}/matmul/clang++/cpu-triv-policy ${CXX_OPTIONS} matmul/cpu-triv-policy.cpp -DA_ROW -DB_ROW -DC_ROW

${BUILD_DIR}/matmul/g++/cpu-poly-noarr: matmul/cpu-poly-noarr.cpp matmul/noarrmain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/g++
	${GCC} -o ${BUILD_DIR}/matmul/g++/cpu-poly-noarr ${CXX_OPTIONS} matmul/cpu-poly-noarr.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_I -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=0 -DDIM_ORDER=1
${BUILD_DIR}/matmul/g++/cpu-poly-noarr-bag: matmul/cpu-poly-noarr-bag.cpp matmul/noarrmain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/g++
	${GCC} -o ${BUILD_DIR}/matmul/g++/cpu-poly-noarr-bag ${CXX_OPTIONS} matmul/cpu-poly-noarr-bag.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_I -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=0 -DDIM_ORDER=1
${BUILD_DIR}/matmul/g++/cpu-poly-policy: matmul/cpu-poly-policy.cpp matmul/policymain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/g++
	${GCC} -o ${BUILD_DIR}/matmul/g++/cpu-poly-policy ${CXX_OPTIONS} matmul/cpu-poly-policy.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_I -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=0 -DDIM_ORDER=1

${BUILD_DIR}/matmul/clang++/cpu-poly-noarr: matmul/cpu-poly-noarr.cpp matmul/noarrmain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/clang++
	${CLANG} -o ${BUILD_DIR}/matmul/clang++/cpu-poly-noarr ${CXX_OPTIONS} matmul/cpu-poly-noarr.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_I -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=0 -DDIM_ORDER=1
${BUILD_DIR}/matmul/clang++/cpu-poly-noarr-bag: matmul/cpu-poly-noarr-bag.cpp matmul/noarrmain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/clang++
	${CLANG} -o ${BUILD_DIR}/matmul/clang++/cpu-poly-noarr-bag ${CXX_OPTIONS} matmul/cpu-poly-noarr-bag.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_I -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=0 -DDIM_ORDER=1
${BUILD_DIR}/matmul/clang++/cpu-poly-policy: matmul/cpu-poly-policy.cpp matmul/policymain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/clang++
	${CLANG} -o ${BUILD_DIR}/matmul/clang++/cpu-poly-policy ${CXX_OPTIONS} matmul/cpu-poly-policy.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_I -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=0 -DDIM_ORDER=1

${BUILD_DIR}/matmul/g++/cpu-poly-noarr2: matmul/cpu-poly-noarr.cpp matmul/noarrmain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/g++
	${GCC} -o ${BUILD_DIR}/matmul/g++/cpu-poly-noarr2 ${CXX_OPTIONS} matmul/cpu-poly-noarr.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_I -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=1 -DDIM_ORDER=1
${BUILD_DIR}/matmul/g++/cpu-poly-noarr-bag2: matmul/cpu-poly-noarr-bag.cpp matmul/noarrmain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/g++
	${GCC} -o ${BUILD_DIR}/matmul/g++/cpu-poly-noarr-bag2 ${CXX_OPTIONS} matmul/cpu-poly-noarr-bag.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_I -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=1 -DDIM_ORDER=1
${BUILD_DIR}/matmul/g++/cpu-poly-policy2: matmul/cpu-poly-policy.cpp matmul/policymain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/g++
	${GCC} -o ${BUILD_DIR}/matmul/g++/cpu-poly-policy2 ${CXX_OPTIONS} matmul/cpu-poly-policy.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_I -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=1 -DDIM_ORDER=1

${BUILD_DIR}/matmul/clang++/cpu-poly-noarr2: matmul/cpu-poly-noarr.cpp matmul/noarrmain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/clang++
	${CLANG} -o ${BUILD_DIR}/matmul/clang++/cpu-poly-noarr2 ${CXX_OPTIONS} matmul/cpu-poly-noarr.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_I -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=1 -DDIM_ORDER=1
${BUILD_DIR}/matmul/clang++/cpu-poly-noarr-bag2: matmul/cpu-poly-noarr-bag.cpp matmul/noarrmain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/clang++
	${CLANG} -o ${BUILD_DIR}/matmul/clang++/cpu-poly-noarr-bag2 ${CXX_OPTIONS} matmul/cpu-poly-noarr-bag.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_I -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=1 -DDIM_ORDER=1
${BUILD_DIR}/matmul/clang++/cpu-poly-policy2: matmul/cpu-poly-policy.cpp matmul/policymain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/clang++
	${CLANG} -o ${BUILD_DIR}/matmul/clang++/cpu-poly-policy2 ${CXX_OPTIONS} matmul/cpu-poly-policy.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_I -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=1 -DDIM_ORDER=1

${BUILD_DIR}/matmul/nvcc/cu-basic-noarr: matmul/cu-basic-noarr.cu matmul/noarrmain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/nvcc
	${NVCC} -o ${BUILD_DIR}/matmul/nvcc/cu-basic-noarr ${CUDA_OPTIONS} matmul/cu-basic-noarr.cu -DA_ROW -DB_ROW -DC_ROW -DBLOCK_SIZE=32
${BUILD_DIR}/matmul/nvcc/cu-basic-noarr-bag: matmul/cu-basic-noarr-bag.cu matmul/noarrmain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/nvcc
	${NVCC} -o ${BUILD_DIR}/matmul/nvcc/cu-basic-noarr-bag ${CUDA_OPTIONS} matmul/cu-basic-noarr-bag.cu -DA_ROW -DB_ROW -DC_ROW -DBLOCK_SIZE=32
${BUILD_DIR}/matmul/nvcc/cu-basic-policy: matmul/cu-basic-policy.cu matmul/policymain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/nvcc
	${NVCC} -o ${BUILD_DIR}/matmul/nvcc/cu-basic-policy ${CUDA_OPTIONS} matmul/cu-basic-policy.cu -DA_ROW -DB_ROW -DC_ROW -DBLOCK_SIZE=32

${BUILD_DIR}/matmul/nvcc/cu-adv-noarr: matmul/cu-adv-noarr.cu matmul/noarrmain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/nvcc
	${NVCC} -o ${BUILD_DIR}/matmul/nvcc/cu-adv-noarr ${CUDA_OPTIONS} matmul/cu-adv-noarr.cu -DA_ROW -DB_ROW -DC_ROW
${BUILD_DIR}/matmul/nvcc/cu-adv-noarr-bag: matmul/cu-adv-noarr-bag.cu matmul/noarrmain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/nvcc
	${NVCC} -o ${BUILD_DIR}/matmul/nvcc/cu-adv-noarr-bag ${CUDA_OPTIONS} matmul/cu-adv-noarr-bag.cu -DA_ROW -DB_ROW -DC_ROW
${BUILD_DIR}/matmul/nvcc/cu-adv-policy: matmul/cu-adv-policy.cu matmul/policymain.hpp
	@mkdir -p ${BUILD_DIR}/matmul/nvcc
	${NVCC} -o ${BUILD_DIR}/matmul/nvcc/cu-adv-policy ${CUDA_OPTIONS} matmul/cu-adv-policy.cu -DA_ROW -DB_ROW -DC_ROW
${BUILD_DIR}/matmul/nvcc/cu-adv-plain: matmul/cu-adv-plain.cu
	@mkdir -p ${BUILD_DIR}/matmul/nvcc
	${NVCC} -o ${BUILD_DIR}/matmul/nvcc/cu-adv-plain ${CUDA_OPTIONS} matmul/cu-adv-plain.cu -DA_ROW -DB_ROW -DC_ROW

histo: noarr-structures \
	${BUILD_DIR}/histo/g++/cpu-all-noarr \
	${BUILD_DIR}/histo/g++/cpu-all-noarr-bag \
	${BUILD_DIR}/histo/g++/cpu-all-plain \
	${BUILD_DIR}/histo/clang++/cpu-all-noarr \
	${BUILD_DIR}/histo/clang++/cpu-all-noarr-bag \
	${BUILD_DIR}/histo/clang++/cpu-all-plain \
	${BUILD_DIR}/histo/nvcc/cu-priv-noarr \
	${BUILD_DIR}/histo/nvcc/cu-priv-noarr-bag \
	${BUILD_DIR}/histo/nvcc/cu-priv-plain

${BUILD_DIR}/histo/g++/cpu-all-noarr: histo/cpu-all-noarr.cpp histo/histomain.hpp
	@mkdir -p ${BUILD_DIR}/histo/g++
	${GCC} -o ${BUILD_DIR}/histo/g++/cpu-all-noarr ${CXX_OPTIONS} histo/cpu-all-noarr.cpp -DHISTO_IMPL=histo_trav_foreach
${BUILD_DIR}/histo/g++/cpu-all-noarr-bag: histo/cpu-all-noarr-bag.cpp histo/histomain.hpp
	@mkdir -p ${BUILD_DIR}/histo/g++
	${GCC} -o ${BUILD_DIR}/histo/g++/cpu-all-noarr-bag ${CXX_OPTIONS} histo/cpu-all-noarr-bag.cpp -DHISTO_IMPL=histo_trav_foreach
${BUILD_DIR}/histo/g++/cpu-all-plain: histo/cpu-all-noarr.cpp histo/histomain.hpp
	@mkdir -p ${BUILD_DIR}/histo/g++
	${GCC} -o ${BUILD_DIR}/histo/g++/cpu-all-plain ${CXX_OPTIONS} histo/cpu-all-noarr.cpp -DHISTO_IMPL=histo_plain_loop

${BUILD_DIR}/histo/clang++/cpu-all-noarr: histo/cpu-all-noarr.cpp histo/histomain.hpp
	@mkdir -p ${BUILD_DIR}/histo/clang++
	${CLANG} -o ${BUILD_DIR}/histo/clang++/cpu-all-noarr ${CXX_OPTIONS} histo/cpu-all-noarr.cpp -DHISTO_IMPL=histo_trav_foreach
${BUILD_DIR}/histo/clang++/cpu-all-noarr-bag: histo/cpu-all-noarr-bag.cpp histo/histomain.hpp
	@mkdir -p ${BUILD_DIR}/histo/clang++
	${CLANG} -o ${BUILD_DIR}/histo/clang++/cpu-all-noarr-bag ${CXX_OPTIONS} histo/cpu-all-noarr-bag.cpp -DHISTO_IMPL=histo_trav_foreach
${BUILD_DIR}/histo/clang++/cpu-all-plain: histo/cpu-all-noarr.cpp histo/histomain.hpp
	@mkdir -p ${BUILD_DIR}/histo/clang++
	${CLANG} -o ${BUILD_DIR}/histo/clang++/cpu-all-plain ${CXX_OPTIONS} histo/cpu-all-noarr.cpp -DHISTO_IMPL=histo_plain_loop

${BUILD_DIR}/histo/nvcc/cu-priv-noarr: histo/cu-priv-noarr.cu histo/histomain.hpp
	@mkdir -p ${BUILD_DIR}/histo/nvcc
	${NVCC} -o ${BUILD_DIR}/histo/nvcc/cu-priv-noarr ${CUDA_OPTIONS} histo/cu-priv-noarr.cu
${BUILD_DIR}/histo/nvcc/cu-priv-noarr-bag: histo/cu-priv-noarr-bag.cu histo/histomain.hpp
	@mkdir -p ${BUILD_DIR}/histo/nvcc
	${NVCC} -o ${BUILD_DIR}/histo/nvcc/cu-priv-noarr-bag ${CUDA_OPTIONS} histo/cu-priv-noarr-bag.cu
${BUILD_DIR}/histo/nvcc/cu-priv-plain: histo/cu-priv-plain.cu
	@mkdir -p ${BUILD_DIR}/histo/nvcc
	${NVCC} -o ${BUILD_DIR}/histo/nvcc/cu-priv-plain ${CUDA_OPTIONS} histo/cu-priv-plain.cu

${BUILD_DIR}/matmul/matrices_64: matmul/gen-matrices.py
	@mkdir -p ${BUILD_DIR}/matmul
	matmul/gen-matrices.py ${BUILD_DIR}/matmul/matrices_64 64

${BUILD_DIR}/matmul/matrices_128: matmul/gen-matrices.py
	@mkdir -p ${BUILD_DIR}/matmul
	matmul/gen-matrices.py ${BUILD_DIR}/matmul/matrices_128 128

${BUILD_DIR}/matmul/matrices_256: matmul/gen-matrices.py
	@mkdir -p ${BUILD_DIR}/matmul
	matmul/gen-matrices.py ${BUILD_DIR}/matmul/matrices_256 256

${BUILD_DIR}/matmul/matrices_512: matmul/gen-matrices.py
	@mkdir -p ${BUILD_DIR}/matmul
	matmul/gen-matrices.py ${BUILD_DIR}/matmul/matrices_512 512

${BUILD_DIR}/matmul/matrices_1024: matmul/gen-matrices.py
	@mkdir -p ${BUILD_DIR}/matmul
	matmul/gen-matrices.py ${BUILD_DIR}/matmul/matrices_1024 1024

${BUILD_DIR}/matmul/matrices_2048: matmul/gen-matrices.py
	@mkdir -p ${BUILD_DIR}/matmul
	matmul/gen-matrices.py ${BUILD_DIR}/matmul/matrices_2048 2048

${BUILD_DIR}/matmul/matrices_4096: matmul/gen-matrices.py
	@mkdir -p ${BUILD_DIR}/matmul
	matmul/gen-matrices.py ${BUILD_DIR}/matmul/matrices_4096 4096

${BUILD_DIR}/matmul/matrices_8192: matmul/gen-matrices.py
	@mkdir -p ${BUILD_DIR}/matmul
	matmul/gen-matrices.py ${BUILD_DIR}/matmul/matrices_8192 8192

${BUILD_DIR}/histo/text: histo/gen-text.sh
	@mkdir -p ${BUILD_DIR}/histo
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

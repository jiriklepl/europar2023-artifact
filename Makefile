GCC := g++
CLANG := clang++
NVCC := nvcc

BUILD_DIR := build

INCLUDE_OPTION := -I noarr-structures/include

CXX_OPTIONS := ${INCLUDE_OPTION} -std=c++20 -Ofast -Wall -DNDEBUG -march=native -mtune=native
CUDA_OPTIONS := ${INCLUDE_OPTION} -std=c++17 -O3 -DNDEBUG --use_fast_math --expt-relaxed-constexpr

.PHONY: all clean matmul histo kmeans generate validate measure

all: matmul histo kmeans

noarr-structures:
	[ -d noarr-structures ] || git clone https://github.com/ParaCoToUl/noarr-structures.git noarr-structures

matmul: noarr-structures \
	${BUILD_DIR}/matmul/gcc/cpu-triv-noarr \
	${BUILD_DIR}/matmul/gcc/cpu-triv-noarr-bag \
	${BUILD_DIR}/matmul/gcc/cpu-triv-policy \
	${BUILD_DIR}/matmul/clang/cpu-triv-noarr \
	${BUILD_DIR}/matmul/clang/cpu-triv-noarr-bag \
	${BUILD_DIR}/matmul/clang/cpu-triv-policy \
	${BUILD_DIR}/matmul/gcc/cpu-poly-noarr \
	${BUILD_DIR}/matmul/gcc/cpu-poly-noarr-bag \
	${BUILD_DIR}/matmul/gcc/cpu-poly-policy \
	${BUILD_DIR}/matmul/clang/cpu-poly-noarr \
	${BUILD_DIR}/matmul/clang/cpu-poly-noarr-bag \
	${BUILD_DIR}/matmul/clang/cpu-poly-policy \
	${BUILD_DIR}/matmul/gcc/cpu-poly-noarr2 \
	${BUILD_DIR}/matmul/gcc/cpu-poly-noarr-bag2 \
	${BUILD_DIR}/matmul/gcc/cpu-poly-policy2 \
	${BUILD_DIR}/matmul/clang/cpu-poly-noarr2 \
	${BUILD_DIR}/matmul/clang/cpu-poly-noarr-bag2 \
	${BUILD_DIR}/matmul/clang/cpu-poly-policy2 \
	${BUILD_DIR}/matmul/nvcc/cu-basic-noarr \
	${BUILD_DIR}/matmul/nvcc/cu-basic-noarr-bag \
	${BUILD_DIR}/matmul/nvcc/cu-basic-policy \
	${BUILD_DIR}/matmul/nvcc/cu-adv-noarr \
	${BUILD_DIR}/matmul/nvcc/cu-adv-noarr-bag \
	${BUILD_DIR}/matmul/nvcc/cu-adv-policy \
	${BUILD_DIR}/matmul/nvcc/cu-adv-plain

${BUILD_DIR}/matmul/gcc/cpu-triv-noarr: matmul/cpu-triv-noarr.cpp
	@mkdir -p ${BUILD_DIR}/matmul/gcc
	${GCC} -o ${BUILD_DIR}/matmul/gcc/cpu-triv-noarr ${CXX_OPTIONS} matmul/cpu-triv-noarr.cpp -DA_ROW -DB_ROW -DC_ROW
${BUILD_DIR}/matmul/gcc/cpu-triv-noarr-bag: matmul/cpu-triv-noarr-bag.cpp
	@mkdir -p ${BUILD_DIR}/matmul/gcc
	${GCC} -o ${BUILD_DIR}/matmul/gcc/cpu-triv-noarr-bag ${CXX_OPTIONS} matmul/cpu-triv-noarr-bag.cpp -DA_ROW -DB_ROW -DC_ROW
${BUILD_DIR}/matmul/gcc/cpu-triv-policy: matmul/cpu-triv-policy.cpp
	@mkdir -p ${BUILD_DIR}/matmul/gcc
	${GCC} -o ${BUILD_DIR}/matmul/gcc/cpu-triv-policy ${CXX_OPTIONS} matmul/cpu-triv-policy.cpp -DA_ROW -DB_ROW -DC_ROW

${BUILD_DIR}/matmul/clang/cpu-triv-noarr: matmul/cpu-triv-noarr.cpp
	@mkdir -p ${BUILD_DIR}/matmul/clang
	${CLANG} -o ${BUILD_DIR}/matmul/clang/cpu-triv-noarr ${CXX_OPTIONS} matmul/cpu-triv-noarr.cpp -DA_ROW -DB_ROW -DC_ROW
${BUILD_DIR}/matmul/clang/cpu-triv-noarr-bag: matmul/cpu-triv-noarr-bag.cpp
	@mkdir -p ${BUILD_DIR}/matmul/clang
	${CLANG} -o ${BUILD_DIR}/matmul/clang/cpu-triv-noarr-bag ${CXX_OPTIONS} matmul/cpu-triv-noarr-bag.cpp -DA_ROW -DB_ROW -DC_ROW
${BUILD_DIR}/matmul/clang/cpu-triv-policy: matmul/cpu-triv-policy.cpp
	@mkdir -p ${BUILD_DIR}/matmul/clang
	${CLANG} -o ${BUILD_DIR}/matmul/clang/cpu-triv-policy ${CXX_OPTIONS} matmul/cpu-triv-policy.cpp -DA_ROW -DB_ROW -DC_ROW

${BUILD_DIR}/matmul/gcc/cpu-poly-noarr: matmul/cpu-poly-noarr.cpp
	@mkdir -p ${BUILD_DIR}/matmul/gcc
	${GCC} -o ${BUILD_DIR}/matmul/gcc/cpu-poly-noarr ${CXX_OPTIONS} matmul/cpu-poly-noarr.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=0 -DDIM_ORDER=1
${BUILD_DIR}/matmul/gcc/cpu-poly-noarr-bag: matmul/cpu-poly-noarr-bag.cpp
	@mkdir -p ${BUILD_DIR}/matmul/gcc
	${GCC} -o ${BUILD_DIR}/matmul/gcc/cpu-poly-noarr-bag ${CXX_OPTIONS} matmul/cpu-poly-noarr-bag.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=0 -DDIM_ORDER=1
${BUILD_DIR}/matmul/gcc/cpu-poly-policy: matmul/cpu-poly-policy.cpp
	@mkdir -p ${BUILD_DIR}/matmul/gcc
	${GCC} -o ${BUILD_DIR}/matmul/gcc/cpu-poly-policy ${CXX_OPTIONS} matmul/cpu-poly-policy.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=0 -DDIM_ORDER=1

${BUILD_DIR}/matmul/clang/cpu-poly-noarr: matmul/cpu-poly-noarr.cpp
	@mkdir -p ${BUILD_DIR}/matmul/clang
	${CLANG} -o ${BUILD_DIR}/matmul/clang/cpu-poly-noarr ${CXX_OPTIONS} matmul/cpu-poly-noarr.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=0 -DDIM_ORDER=1
${BUILD_DIR}/matmul/clang/cpu-poly-noarr-bag: matmul/cpu-poly-noarr-bag.cpp
	@mkdir -p ${BUILD_DIR}/matmul/clang
	${CLANG} -o ${BUILD_DIR}/matmul/clang/cpu-poly-noarr-bag ${CXX_OPTIONS} matmul/cpu-poly-noarr-bag.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=0 -DDIM_ORDER=1
${BUILD_DIR}/matmul/clang/cpu-poly-policy: matmul/cpu-poly-policy.cpp
	@mkdir -p ${BUILD_DIR}/matmul/clang
	${CLANG} -o ${BUILD_DIR}/matmul/clang/cpu-poly-policy ${CXX_OPTIONS} matmul/cpu-poly-policy.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=16 -DBLOCK_ORDER=0 -DDIM_ORDER=1

${BUILD_DIR}/matmul/gcc/cpu-poly-noarr2: matmul/cpu-poly-noarr.cpp
	@mkdir -p ${BUILD_DIR}/matmul/gcc
	${GCC} -o ${BUILD_DIR}/matmul/gcc/cpu-poly-noarr2 ${CXX_OPTIONS} matmul/cpu-poly-noarr.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=32 -DBLOCK_ORDER=5 -DDIM_ORDER=1
${BUILD_DIR}/matmul/gcc/cpu-poly-noarr-bag2: matmul/cpu-poly-noarr-bag.cpp
	@mkdir -p ${BUILD_DIR}/matmul/gcc
	${GCC} -o ${BUILD_DIR}/matmul/gcc/cpu-poly-noarr-bag2 ${CXX_OPTIONS} matmul/cpu-poly-noarr-bag.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=32 -DBLOCK_ORDER=5 -DDIM_ORDER=1
${BUILD_DIR}/matmul/gcc/cpu-poly-policy2: matmul/cpu-poly-policy.cpp
	@mkdir -p ${BUILD_DIR}/matmul/gcc
	${GCC} -o ${BUILD_DIR}/matmul/gcc/cpu-poly-policy2 ${CXX_OPTIONS} matmul/cpu-poly-policy.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=32 -DBLOCK_ORDER=5 -DDIM_ORDER=1

${BUILD_DIR}/matmul/clang/cpu-poly-noarr2: matmul/cpu-poly-noarr.cpp
	@mkdir -p ${BUILD_DIR}/matmul/clang
	${CLANG} -o ${BUILD_DIR}/matmul/clang/cpu-poly-noarr2 ${CXX_OPTIONS} matmul/cpu-poly-noarr.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=32 -DBLOCK_ORDER=5 -DDIM_ORDER=1
${BUILD_DIR}/matmul/clang/cpu-poly-noarr-bag2: matmul/cpu-poly-noarr-bag.cpp
	@mkdir -p ${BUILD_DIR}/matmul/clang
	${CLANG} -o ${BUILD_DIR}/matmul/clang/cpu-poly-noarr-bag2 ${CXX_OPTIONS} matmul/cpu-poly-noarr-bag.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=32 -DBLOCK_ORDER=5 -DDIM_ORDER=1
${BUILD_DIR}/matmul/clang/cpu-poly-policy2: matmul/cpu-poly-policy.cpp
	@mkdir -p ${BUILD_DIR}/matmul/clang
	${CLANG} -o ${BUILD_DIR}/matmul/clang/cpu-poly-policy2 ${CXX_OPTIONS} matmul/cpu-poly-policy.cpp -DA_ROW -DB_ROW -DC_ROW -DBLOCK_J -DBLOCK_K -DBLOCK_SIZE=32 -DBLOCK_ORDER=5 -DDIM_ORDER=1

${BUILD_DIR}/matmul/nvcc/cu-basic-noarr: matmul/cu-basic-noarr.cu
	@mkdir -p ${BUILD_DIR}/matmul/nvcc
	${NVCC} -o ${BUILD_DIR}/matmul/nvcc/cu-basic-noarr ${CUDA_OPTIONS} matmul/cu-basic-noarr.cu -DA_ROW -DB_ROW -DC_ROW -DBLOCK_SIZE=32
${BUILD_DIR}/matmul/nvcc/cu-basic-noarr-bag: matmul/cu-basic-noarr-bag.cu
	@mkdir -p ${BUILD_DIR}/matmul/nvcc
	${NVCC} -o ${BUILD_DIR}/matmul/nvcc/cu-basic-noarr-bag ${CUDA_OPTIONS} matmul/cu-basic-noarr-bag.cu -DA_ROW -DB_ROW -DC_ROW -DBLOCK_SIZE=32
${BUILD_DIR}/matmul/nvcc/cu-basic-policy: matmul/cu-basic-policy.cu
	@mkdir -p ${BUILD_DIR}/matmul/nvcc
	${NVCC} -o ${BUILD_DIR}/matmul/nvcc/cu-basic-policy ${CUDA_OPTIONS} matmul/cu-basic-policy.cu -DA_ROW -DB_ROW -DC_ROW -DBLOCK_SIZE=32

${BUILD_DIR}/matmul/nvcc/cu-adv-noarr: matmul/cu-adv-noarr.cu
	@mkdir -p ${BUILD_DIR}/matmul/nvcc
	${NVCC} -o ${BUILD_DIR}/matmul/nvcc/cu-adv-noarr ${CUDA_OPTIONS} matmul/cu-adv-noarr.cu -DA_ROW -DB_ROW -DC_ROW
${BUILD_DIR}/matmul/nvcc/cu-adv-noarr-bag: matmul/cu-adv-noarr-bag.cu
	@mkdir -p ${BUILD_DIR}/matmul/nvcc
	${NVCC} -o ${BUILD_DIR}/matmul/nvcc/cu-adv-noarr-bag ${CUDA_OPTIONS} matmul/cu-adv-noarr-bag.cu -DA_ROW -DB_ROW -DC_ROW
${BUILD_DIR}/matmul/nvcc/cu-adv-policy: matmul/cu-adv-policy.cu
	@mkdir -p ${BUILD_DIR}/matmul/nvcc
	${NVCC} -o ${BUILD_DIR}/matmul/nvcc/cu-adv-policy ${CUDA_OPTIONS} matmul/cu-adv-policy.cu -DA_ROW -DB_ROW -DC_ROW
${BUILD_DIR}/matmul/nvcc/cu-adv-plain: matmul/cu-adv-plain.cu
	@mkdir -p ${BUILD_DIR}/matmul/nvcc
	${NVCC} -o ${BUILD_DIR}/matmul/nvcc/cu-adv-plain ${CUDA_OPTIONS} matmul/cu-adv-plain.cu -DA_ROW -DB_ROW -DC_ROW

histo: noarr-structures \
	${BUILD_DIR}/histo/gcc/cpu-noarr \
	${BUILD_DIR}/histo/gcc/cpu-noarr-bag \
	${BUILD_DIR}/histo/gcc/cpu-plain \
	${BUILD_DIR}/histo/clang/cpu-noarr \
	${BUILD_DIR}/histo/clang/cpu-noarr-bag \
	${BUILD_DIR}/histo/clang/cpu-plain \
	${BUILD_DIR}/histo/nvcc/cu-noarr \
	${BUILD_DIR}/histo/nvcc/cu-noarr-bag \
	${BUILD_DIR}/histo/nvcc/cu-plain

${BUILD_DIR}/histo/gcc/cpu-noarr: histo/cpu-noarr.cpp
	@mkdir -p ${BUILD_DIR}/histo/gcc
	${GCC} -o ${BUILD_DIR}/histo/gcc/cpu-noarr ${CXX_OPTIONS} histo/cpu-noarr.cpp -DHISTO_IMPL=histo_trav_foreach
${BUILD_DIR}/histo/gcc/cpu-noarr-bag: histo/cpu-noarr-bag.cpp
	@mkdir -p ${BUILD_DIR}/histo/gcc
	${GCC} -o ${BUILD_DIR}/histo/gcc/cpu-noarr-bag ${CXX_OPTIONS} histo/cpu-noarr-bag.cpp -DHISTO_IMPL=histo_trav_foreach
${BUILD_DIR}/histo/gcc/cpu-plain: histo/cpu-noarr.cpp
	@mkdir -p ${BUILD_DIR}/histo/gcc
	${GCC} -o ${BUILD_DIR}/histo/gcc/cpu-plain ${CXX_OPTIONS} histo/cpu-noarr.cpp -DHISTO_IMPL=histo_plain_loop

${BUILD_DIR}/histo/clang/cpu-noarr: histo/cpu-noarr.cpp
	@mkdir -p ${BUILD_DIR}/histo/clang
	${CLANG} -o ${BUILD_DIR}/histo/clang/cpu-noarr ${CXX_OPTIONS} histo/cpu-noarr.cpp -DHISTO_IMPL=histo_trav_foreach
${BUILD_DIR}/histo/clang/cpu-noarr-bag: histo/cpu-noarr-bag.cpp
	@mkdir -p ${BUILD_DIR}/histo/clang
	${CLANG} -o ${BUILD_DIR}/histo/clang/cpu-noarr-bag ${CXX_OPTIONS} histo/cpu-noarr-bag.cpp -DHISTO_IMPL=histo_trav_foreach
${BUILD_DIR}/histo/clang/cpu-plain: histo/cpu-noarr.cpp
	@mkdir -p ${BUILD_DIR}/histo/clang
	${CLANG} -o ${BUILD_DIR}/histo/clang/cpu-plain ${CXX_OPTIONS} histo/cpu-noarr.cpp -DHISTO_IMPL=histo_plain_loop

${BUILD_DIR}/histo/nvcc/cu-noarr: histo/cu-noarr.cu
	@mkdir -p ${BUILD_DIR}/histo/nvcc
	${NVCC} -o ${BUILD_DIR}/histo/nvcc/cu-noarr ${CUDA_OPTIONS} histo/cu-noarr.cu
${BUILD_DIR}/histo/nvcc/cu-noarr-bag: histo/cu-noarr-bag.cu
	@mkdir -p ${BUILD_DIR}/histo/nvcc
	${NVCC} -o ${BUILD_DIR}/histo/nvcc/cu-noarr-bag ${CUDA_OPTIONS} histo/cu-noarr-bag.cu
${BUILD_DIR}/histo/nvcc/cu-plain: histo/cu-plain.cu
	@mkdir -p ${BUILD_DIR}/histo/nvcc
	${NVCC} -o ${BUILD_DIR}/histo/nvcc/cu-plain ${CUDA_OPTIONS} histo/cu-plain.cu

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

validate:

kmeans: noarr-structures 

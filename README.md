# Artifact Submission: Noarr Traversers

This is a replication package containing code and experimental results related to a *Euro-Par 2023* paper titled: *Pure C++ Approach to Optimized Parallel~Traversal of Regular Data~Structures*

The [Overview](#overview) section gives a summary of all relevant source files, test input generators, and test scripts. The [Build](#build) section then goes through the build requirements and build instructions for the experimental implementations; and, finally, the [Experiments](#experiments) section provides the specifics about experiment methodology and data visualization.

The last section, [Bonus](#bonus) then showcases some implementations that should show the viability of our approach in algorithms outside linear algebra; and they also present more advanced uses of the Noarr library.

For a better look at the Noarr library itself, see [https://github.com/ParaCoToUl/noarr-structures/](https://github.com/ParaCoToUl/noarr-structures/) (the main repository for the library) or [https://link.springer.com/chapter/10.1007/978-3-031-22677-9_27](https://link.springer.com/chapter/10.1007/978-3-031-22677-9_27) (the paper associated with the initial proposal of the Noarr abstraction).

## Overview

The artifact has the following file structure.

- [Makefile](Makefille): Contains build instructions for all the experimental implementations and their inputs

  - [Build](#build) section of te artifact specifies its use

- [matmul](matmul): The directory containing all matrix multiplication implementations used in the measurements.

  - Files beginning with either `cu-`, or `cpu-`: The implementations of the matrix multiplication algorithm, each file contains just the part that defines and executes the execution.

  Each such implementation has up to 4 versions (distinguished by the last word before the file extension; for each implementation, two of these are written using our propossed Noarr abstraction - with Noarr bags or without; others serve as the pure-C++ baseline) - each group of corresponding different versions is programmed to perform the exact same sequence of memory instructions (before any compiler optimizations).

  The sources `cpu-blk-*.cpp` and `cpu-triv-*.cpp` use the `LOG` macro (defined in a corresponding `*main.hpp` file) for debugging purposes. The macro uses also serve as comments explaining the algorithm (the preprocessor ignores the macro argument unless `LOGGING` is defined).

  - [noarrmain.cpp](matmul/noarrmain.hpp), [policymain.hpp](matmul/policymain.hpp), [plainymain.hpp](matmul/plainymain.hpp): Header files that define memory management, IO, and time measurement.

  - [gen-matrices.py](matmul/gen-matrices.py): A Python script used to prepare input matrices with random content.

- [histo](histo): The directory containing all histogram implementations used in the measurements.

  - Files beginning with either `cu-`, or `cpu-`: The implementations of the histogram algorithm, each file contains just the part that defines and executes the execution.

  Each such implementation has 3 versions (distinguished by the last word before the file extension; for each implementation, two of these are written using our propossed Noarr abstraction - with Noarr bags or without; others serve as the pure-C++ baseline) - each group of corresponding different versions is programmed to perform the exact same sequence of memory instructions (before any compiler optimizations).

  - [histomain.cpp](histo/histomain.hpp): Header file that defines memory management, IO, and time measurement.

  - [gen-matrices.py](matmul/gen-text.sh): A shell script used to generate a random text with uniformly distributed characters.

- [matmul-measure.sh](matmul-measure.sh) and [histo-measure.sh](histo-measure.sh): Shell scripts used to measure the runtimes for matrix multiplication and histogram, respectively.

- [matmul-validate.sh](matmul-validate.sh) and [histo-validate.sh](histo-validate.sh): Shell scripts used to test whether all the corresponding implementations of the given algorithm produce the same result, given the same input.

- [matmul-blk-test.sh](matmul-blk-test.sh) and [matmul-triv-test.sh](matmul-triv-test.sh): Debugging scripts that compare the work process of the `cpu-blk-*` and `cpu-triv-*` implementations, respectively, with many different configurations. The comparison is performed via explicit logging of arithmetic and memory operations.

## Build

This section is dedicated to build requirements and build instructions.

### Requirements

All CPU implementations require a C++20-compliant compiler and all CUDA implementations require a C++17-compliant version of the *nvcc* compiler. The compilers are also expected to support all options specified in [Makefile](Makefile), but `-O3` should be sufficient to get close-enough results.

Apart from the [noarr-structutres](https://github.com/ParaCoToUl/noarr-structures.git) library, the implementations themselves have no external dependencies other than *TBB* (but that is limited to just 2 implementations; use [NoTBB/Makefile](NoTBB/Makefile) to skip the TBB implementations).

**Other requirements:**

For generating the input files, *Python* is required along with the following *Python* packages (in as current versions as possible - as of 3-3-2023):

- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`

### Build instructions

`make all` builds all matrix multiplication, histogram and kmeans examples. All compilation should proceed with no warnings if all the requirements are met.

- replace `all` with `matmul`, `histo` or `kmeans`, to build the implementations of only one algorithm.

`make all CLANG=@: GCC=@:` builds all matrix multiplication and histogram CUDA examples.

`make all NVCC=@:` builds all matrix multiplication and histogram C++ examples using the *clang++* and *g++* compilers (you can also add `CLANG=@:` or `GCC=@:` to prevent the use of the respective compiler; or you can override one of them to use a preferred C++ compiler).

`make generate` generates matrix multiplication input files with sizes ranging from `2**10` to `2**13`; and the random text for th histogram algorithm.

`make generate-small` generates matrix multiplication input files with sizes ranging from `2**6` to `2**10`

## Experiments

The git branches `gpulab` and `parlab`, each, contain the `./run_job.sh` used to perform the experiments on a particular machine in a Slurm-managed cluster -- measuring GPU implementations and CPU implementations, respectively.

Both branches also contain `results.tar.gz` with experimental results stored in the CSV format.

### Visualization

The measured data are visualized using plots generated by [matmul.R](matmul.R) and [histo.R](histo.R) R-scripts that are written using the *ggplot2*, *dplyr* and *stringr* libraries.

`make plots` runs both of the R scripts.

## Bonus

This section contains what didn't fit the paper but might be still interesting to explore

### Kmeans

Noarr is not viable just in simple computation kernels performing work on some regular structures. The implementation of this algorithm showcases some of the more advanced uses of the library, including a simple data (de)serialization example.

- [kmeans](kmeans) contains three versions of an implementation of the kmeans algorithm (two Noarr versions and one in pure C++). One Noarr version uses Noarr bags and the other one does not. The names follow the naming scheme of the other two algorithms.

  - [kmeans-gen.py](kmeans/kmeans-gen.py): A Python script that generates data that are very likely to be clustered deterministically into the expected number of clusters. It also attempts to choose inputs that make the process more resistant to numeric errors caused by floating-point arithmetics.

  `make generate-kmeans` uses this script to generate few test inputs (it uses a pre-specified seed to promote replicability).

The kmeans experiments can be performed by the following process:

```sh
make kmeans
make generate-kmeans

./kmeans-validate.sh && ./kmens-measure.sh
```

This experiment also includes comparison against the `sklearn` kmeans algorithm configured to perform the same job. It is included mainly to provide the correct output.

### Stencil

This algorithm does not have an experiment associated with it as the required abstractions are not yet fully optimized, but it is included to showcase the flexibility of the presented abstractions.

- [stencil](stencil): This directory, as usual, contains three versions of a simple implementation of a blur stencil (performing flat blur with a specified neighborhood radius); two versions that use the Noarr abstraction (one with Noarr bags and the other without), one that is written in pure C++.

  - [draw_image.py](stencil/gen_image.py): Generates an image represented as a sequence of `height * width * channels` numbers ranging from 0 to 256. The image consists of a noisy rectangle in the middle of a black field.

  - [draw_image.py](stencil/draw_image.py): Transforms a sequence of `height * width * channels` numbers into an image. (If it can be interpreted as one.)

  This is included just to help visualize that the algorithm performs the expected work.

**Build:**

```sh
# alternatively, link it, if it is already cloned elsewhere
git clone https://github.com/ParaCoToUl/noarr-structures/

mkdir -p build/stencil/

FAVORITE_COMPILER -std=c++20 -Ofast -Wall -DNDEBUG \
    -o build/stencil/noarr -Inoarr-structures stencil/cpu-stencil-noarr.cpp

FAVORITE_COMPILER -std=c++20 -Ofast -Wall -DNDEBUG \
    -o build/stencil/noarr-bag -Inoarr-structures stencil/cpu-stencil-noarr.cpp

FAVORITE_COMPILER -std=c++20 -Ofast -Wall -DNDEBUG \
    -o build/stencil/plain -Inoarr-structures stencil/cpu-stencil-noarr.cpp
```

**Run:**

```sh
build/stencil/BINARY_NAME <input> <height> <width> <channels> <radius>
```
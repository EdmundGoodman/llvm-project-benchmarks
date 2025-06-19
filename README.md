# Benchmarking LLVM MLIR

At the 2024 European LLVM Developers' Meeting, Mehdi Amini and Jeff Nui
presented their keynote talk "How Slow is MLIR" [^1] [^2]. This presented
microbenchmarks for key operations in the MLIR compiler [^3], such as traversing the
IR and creating operations. However, the code for the benchmarks and
instructions for running them are not easily available online. This repo
consolidates the benchmark code with instructions for running it in a single
easier-to-find location.

## Finding the benchmarks

The benchmarks discussed in this talk are available on a branch of Mehdi's fork of
LLVM [available here](https://github.com/joker-eph/llvm-project/tree/benchmarks),
with a diff with the main branch
[available here](https://github.com/llvm/llvm-project/compare/main...joker-eph:llvm-project:benchmarks).

## Running the benchmarks

To run the benchmarks, first clone the repository at the correct branch:

```bash
https://github.com/EdmundGoodman/llvm-project-benchmarks/
```

Then, build the appropriate CMake target:

```bash
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DLLVM_ENABLE_BENCHMARKS=ON \
   -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build . --target MLIR_IR_Benchmark
```

The benchmarks can then be run with:

```bash
./tools/mlir/unittests/Benchmarks/MLIR_IR_Benchmark
```

These benchmarks are implemented using the
[Google microbenchmark support library](https://github.com/google/benchmark/tree/main)
and as such can be configured with the command line interface:

```text
benchmark [--benchmark_list_tests={true|false}]
          [--benchmark_filter=<regex>]
          [--benchmark_min_time=`<integer>x` OR `<float>s` ]
          [--benchmark_min_warmup_time=<min_warmup_time>]
          [--benchmark_repetitions=<num_repetitions>]
          [--benchmark_enable_random_interleaving={true|false}]
          [--benchmark_report_aggregates_only={true|false}]
          [--benchmark_display_aggregates_only={true|false}]
          [--benchmark_format=<console|json|csv>]
          [--benchmark_out=<filename>]
          [--benchmark_out_format=<json|console|csv>]
          [--benchmark_color={auto|true|false}]
          [--benchmark_counters_tabular={true|false}]
          [--benchmark_context=<key>=<value>,...]
          [--benchmark_time_unit={ns|us|ms|s}]
          [--v=<verbosity>]
```

[^1]: <https://www.youtube.com/watch?v=7qvVMUSxqz4>
[^2]: <https://llvm.org/devmtg/2024-04/slides/Keynote/Amini-Niu-HowSlowIsMLIR.pdf>
[^3]: <https://mlir.llvm.org/>

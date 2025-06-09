//===- ConstantFolding.cpp - Benchmark  ---------------------------------- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "TestBenchDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <memory>
#include <unistd.h>
#include <random>

#include <valgrind/callgrind.h>
#include "benchmark/benchmark.h"

using namespace mlir;
void mlirBenchmarkInitLLVM(int argc, const char **argv);


class Base {
public:
    // Inline function - inlined by the compiler
    inline int inlineFunction(int a, int b) {
        return a - b;
    }

    // Static function - resolved at compile time
    static int staticFunction(int a, int b) {
        return a - b;
    }

    // Virtual function - resolved at runtime via vtable
    virtual int virtualFunction(int a, int b) {
        return a - b;
    }

    // Non-virtual instance method for comparison
    __attribute__((noinline))
    int regularFunctionNoinline(int a, int b) {
        return a - b;
    }

    // Non-virtual instance method for comparison
    int regularFunction(int a, int b) {
        return a - b;
    }

    // virtual ~Base() = default;
};

// Derived class that overrides the virtual function
class Derived : public Base {
public:
    int virtualFunction(int a, int b) override {
        return b - a;
    }
};


namespace {
class Dynamism : public benchmark::Fixture {
public:
    void SetUp(::benchmark::State &state) final {
        const char *cmd = "bench";
        const char **argv = &cmd;
        int argc = 1;
        // Init LLVM to get backtraces on crash
        mlirBenchmarkInitLLVM(argc, argv);
        ctx = std::make_unique<MLIRContext>();
        ctx->allowUnregisteredDialects();
        unknownLoc = UnknownLoc::get(ctx.get());
        moduleOp = OpBuilder(ctx.get()).create<ModuleOp>(unknownLoc);

        a = 5;
        b = 7;
        result = 0;
        monomorphicPtr = &baseObj;
        polymorphicPtr = &derivedObj;
    }

    void TearDown(::benchmark::State &state) final {
        moduleOp.release()->erase();
        ctx.reset();
    }

    std::unique_ptr<MLIRContext> ctx;
    OwningOpRef<ModuleOp> moduleOp;
    UnknownLoc unknownLoc;

    int a, b, result;
    Base baseObj;
    Derived derivedObj;
    Base* monomorphicPtr;
    Base* polymorphicPtr;

};
} // namespace


// No function call
BENCHMARK_DEFINE_F(Dynamism, noCall)
(benchmark::State &state) {
    // state.PauseTiming();
    CALLGRIND_START_INSTRUMENTATION;
    CALLGRIND_ZERO_STATS;
    result = a - b;
    benchmark::DoNotOptimize(result);
    CALLGRIND_STOP_INSTRUMENTATION;
    CALLGRIND_DUMP_STATS;

    // // state.ResumeTiming();
    for (auto _ : state) {
    // for (int j = 0; j < state.range(0); ++j) {
    //     result = a - b;
    //     benchmark::DoNotOptimize(result);
    // }
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(Dynamism, noCall)
    ->Ranges({{10, 10 * 1000}}) // * 1000}})
    ->Complexity(benchmark::oN);

// // Inline function call
// BENCHMARK_DEFINE_F(Dynamism, inlineCall)
// (benchmark::State &state) {
//     CALLGRIND_START_INSTRUMENTATION;
//     CALLGRIND_ZERO_STATS;
//     result = baseObj.inlineFunction(a, b);
//     benchmark::DoNotOptimize(result);
//     CALLGRIND_STOP_INSTRUMENTATION;
//     CALLGRIND_DUMP_STATS;

//     for (auto _ : state) {
//     for (int j = 0; j < state.range(0); ++j) {
//         result = baseObj.inlineFunction(a, b);
//         benchmark::DoNotOptimize(result);
//     }
//     }
//     state.SetComplexityN(state.range(0));
// }
// BENCHMARK_REGISTER_F(Dynamism, inlineCall)
//     ->Ranges({{10, 10 * 1000 * 1000}})
//     ->Complexity(benchmark::oN);

// // Static function call
// BENCHMARK_DEFINE_F(Dynamism, staticCall)
// (benchmark::State &state) {
//     CALLGRIND_START_INSTRUMENTATION;
//     CALLGRIND_ZERO_STATS;
//     result = Base::staticFunction(a, b);
//     benchmark::DoNotOptimize(result);
//     CALLGRIND_STOP_INSTRUMENTATION;
//     CALLGRIND_DUMP_STATS;

//     for (auto _ : state) {
//     for (int j = 0; j < state.range(0); ++j) {
//         result = Base::staticFunction(a, b);
//         benchmark::DoNotOptimize(result);
//     }
//     }
//     state.SetComplexityN(state.range(0));
// }
// BENCHMARK_REGISTER_F(Dynamism, staticCall)
//     ->Ranges({{10, 10 * 1000 * 1000}})
//     ->Complexity(benchmark::oN);

// Regular function call
BENCHMARK_DEFINE_F(Dynamism, regularCall)
(benchmark::State &state) {
    state.PauseTiming();
    // CALLGRIND_START_INSTRUMENTATION;
    // CALLGRIND_ZERO_STATS;
    // result = baseObj.regularFunction(a, b);
    // benchmark::DoNotOptimize(result);
    // CALLGRIND_STOP_INSTRUMENTATION;
    // CALLGRIND_DUMP_STATS;

    // // state.ResumeTiming();
    for (auto _ : state) {
    // for (int j = 0; j < state.range(0); ++j) {
    //     result = baseObj.regularFunction(a, b);
    //     benchmark::DoNotOptimize(result);
    // }
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(Dynamism, regularCall)
    ->Ranges({{10, 10 * 1000}}) // * 1000}})
    ->Complexity(benchmark::oN);

// Regular function call
BENCHMARK_DEFINE_F(Dynamism, regularCallNoinline)
(benchmark::State &state) {
    // state.PauseTiming();
    CALLGRIND_START_INSTRUMENTATION;
    CALLGRIND_ZERO_STATS;
    result = baseObj.regularFunction(a, b);
    benchmark::DoNotOptimize(result);
    CALLGRIND_STOP_INSTRUMENTATION;
    CALLGRIND_DUMP_STATS;

    // // state.ResumeTiming();
    for (auto _ : state) {
    // for (int j = 0; j < state.range(0); ++j) {
    //     result = baseObj.regularFunctionNoinline(a, b);
    //     benchmark::DoNotOptimize(result);
    // }
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(Dynamism, regularCallNoinline)
    ->Ranges({{10, 10 * 1000}}) // * 1000}})
    ->Complexity(benchmark::oN);

// Regular pointer function call
BENCHMARK_DEFINE_F(Dynamism, regularPointerCall)
(benchmark::State &state) {
    // state.PauseTiming();
    CALLGRIND_START_INSTRUMENTATION;
    CALLGRIND_ZERO_STATS;
    result = baseObj.regularFunction(a, b);
    benchmark::DoNotOptimize(result);
    CALLGRIND_STOP_INSTRUMENTATION;
    CALLGRIND_DUMP_STATS;

    // // state.ResumeTiming();
    for (auto _ : state) {
    // for (int j = 0; j < state.range(0); ++j) {
    //     result = monomorphicPtr->regularFunction(a, b);
    //     benchmark::DoNotOptimize(result);
    // }
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(Dynamism, regularPointerCall)
    ->Ranges({{10, 10 * 1000}}) // * 1000}})
    ->Complexity(benchmark::oN);

// // Monomorphic virtual function call
// BENCHMARK_DEFINE_F(Dynamism, monomorphicCall)
// (benchmark::State &state) {
//     CALLGRIND_START_INSTRUMENTATION;
//     CALLGRIND_ZERO_STATS;
//     result = baseObj.virtualFunction(a, b);
//     benchmark::DoNotOptimize(result);
//     CALLGRIND_STOP_INSTRUMENTATION;
//     CALLGRIND_DUMP_STATS;

//     for (auto _ : state) {
//     for (int j = 0; j < state.range(0); ++j) {
//         result = baseObj.virtualFunction(a, b);
//         benchmark::DoNotOptimize(result);
//     }
//     }
//     state.SetComplexityN(state.range(0));
// }
// BENCHMARK_REGISTER_F(Dynamism, monomorphicCall)
//     ->Ranges({{10, 10 * 1000 * 1000}})
//     ->Complexity(benchmark::oN);

// // Polymorphic virtual function call
// BENCHMARK_DEFINE_F(Dynamism, polymorphicCall)
// (benchmark::State &state) {
//     CALLGRIND_START_INSTRUMENTATION;
//     CALLGRIND_ZERO_STATS;
//     result = derivedObj.virtualFunction(a, b);
//     benchmark::DoNotOptimize(result);
//     CALLGRIND_STOP_INSTRUMENTATION;
//     CALLGRIND_DUMP_STATS;

//     for (auto _ : state) {
//     for (int j = 0; j < state.range(0); ++j) {
//         result = derivedObj.virtualFunction(a, b);
//         benchmark::DoNotOptimize(result);
//     }
//     }
//     state.SetComplexityN(state.range(0));
// }
// BENCHMARK_REGISTER_F(Dynamism, polymorphicCall)
//     ->Ranges({{10, 10 * 1000 * 1000}})
//     ->Complexity(benchmark::oN);


// // Monomorphic pointer virtual function call
// BENCHMARK_DEFINE_F(Dynamism, monomorphicPointerCall)
// (benchmark::State &state) {
//     CALLGRIND_START_INSTRUMENTATION;
//     CALLGRIND_ZERO_STATS;
//     result = monomorphicPtr->virtualFunction(a, b);
//     benchmark::DoNotOptimize(result);
//     CALLGRIND_STOP_INSTRUMENTATION;
//     CALLGRIND_DUMP_STATS;

//     for (auto _ : state) {
//     for (int j = 0; j < state.range(0); ++j) {
//         result = monomorphicPtr->virtualFunction(a, b);
//         benchmark::DoNotOptimize(result);
//     }
//     }
//     state.SetComplexityN(state.range(0));
// }
// BENCHMARK_REGISTER_F(Dynamism, monomorphicPointerCall)
//     ->Ranges({{10, 10 * 1000 * 1000}})
//     ->Complexity(benchmark::oN);

// // Polymorphic pointer virtual function call
// BENCHMARK_DEFINE_F(Dynamism, polymorphicPointerCall)
// (benchmark::State &state) {
//     CALLGRIND_START_INSTRUMENTATION;
//     CALLGRIND_ZERO_STATS;
//     result = polymorphicPtr->virtualFunction(a, b);
//     benchmark::DoNotOptimize(result);
//     CALLGRIND_STOP_INSTRUMENTATION;
//     CALLGRIND_DUMP_STATS;

//     for (auto _ : state) {
//     for (int j = 0; j < state.range(0); ++j) {
//         result = polymorphicPtr->virtualFunction(a, b);
//         benchmark::DoNotOptimize(result);
//     }
//     }
//     state.SetComplexityN(state.range(0));
// }
// BENCHMARK_REGISTER_F(Dynamism, polymorphicPointerCall)
//     ->Ranges({{10, 10 * 1000 * 1000}})
//     ->Complexity(benchmark::oN);

// Polymorphic pointer virtual function call
BENCHMARK_DEFINE_F(Dynamism, runtimePolymorphicPointerCall)
(benchmark::State &state) {
    state.PauseTiming();
    // CALLGRIND_START_INSTRUMENTATION;
    // CALLGRIND_ZERO_STATS;
    // Base* objPtr;
    // if (state.range(0) < 0) {
    //     objPtr = &baseObj;
    // } else {
    //     objPtr = &derivedObj;
    // }
    // result = objPtr->virtualFunction(a, b);
    // benchmark::DoNotOptimize(result);
    // CALLGRIND_STOP_INSTRUMENTATION;
    // CALLGRIND_DUMP_STATS;

    for (auto _ : state) {
    // // state.PauseTiming();
    // Base* objPtr;
    // if (state.range(0) < 0) {
    //     objPtr = &baseObj;
    // } else {
    //     objPtr = &derivedObj;
    // }
    // // state.ResumeTiming();
    // for (int j = 0; j < state.range(0); ++j) {
    //     result = objPtr->virtualFunction(a, b);
    //     benchmark::DoNotOptimize(result);
    // }
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(Dynamism, runtimePolymorphicPointerCall)
    ->Ranges({{10, 10 * 1000}}) // * 1000}})
    ->Complexity(benchmark::oN);



// // Dynamism on hasTrait
// BENCHMARK_DEFINE_F(Dynamism, hasTrait)
// (benchmark::State &state) {
//     ctx->loadDialect<TestBenchDialect>();
//     OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
//     SmallVector<Operation *> ops;
//     for (int j = 0; j < state.range(0); ++j) {
//         ops.push_back(b.create<OpWithRegion>(unknownLoc));
//     }

//     CALLGRIND_START_INSTRUMENTATION;
//     CALLGRIND_ZERO_STATS;
//     bool hasTrait = ops[0]->hasTrait<OpTrait::SingleBlock>();
//     benchmark::DoNotOptimize(&hasTrait);
//     CALLGRIND_STOP_INSTRUMENTATION;
//     CALLGRIND_DUMP_STATS;

//     for (auto _ : state) {
//     for (Operation *op : ops) {
//         state.ResumeTiming();
//         bool hasTrait = op->hasTrait<OpTrait::SingleBlock>();
//         state.PauseTiming();
//         benchmark::DoNotOptimize(&hasTrait);
//     };
//     }
//     state.SetComplexityN(state.range(0));
// }
// BENCHMARK_REGISTER_F(Dynamism, hasTrait)
//     ->Ranges({{10, 10 * 1000 * 1000}})
//     ->Complexity(benchmark::oN);

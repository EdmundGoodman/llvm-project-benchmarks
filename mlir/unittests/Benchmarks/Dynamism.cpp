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

#include "benchmark/benchmark.h"

using namespace mlir;
void mlirBenchmarkInitLLVM(int argc, const char **argv);




class Base {
public:
    // Static function - resolved at compile time
    static int staticFunction(int a, int b) {
        return a + b;
    }

    // Virtual function - resolved at runtime via vtable
    virtual int virtualFunction(int a, int b) {
        return a + b;
    }

    // Non-virtual instance method for comparison
    int regularFunction(int a, int b) {
        return a + b;
    }

    virtual ~Base() = default;
};

// Derived class that overrides the virtual function
class Derived : public Base {
public:
    int virtualFunction(int a, int b) override {
        return a * b; // Different implementation
    }
};

// Helper function to prevent compiler optimization from eliminating our calls
template<typename T>
void doNotOptimize(T&& value) {
    // This volatile prevents the compiler from optimizing away our function calls
    volatile auto temp = value;
    (void)temp; // Suppress unused variable warning
}



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



// Static function call
BENCHMARK_DEFINE_F(Dynamism, staticCall)(benchmark::State &state) {
    for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j) {
        result = Base::staticFunction(a, b);
        doNotOptimize(result);
    }
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(Dynamism, staticCall)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

// Regular function call
BENCHMARK_DEFINE_F(Dynamism, regularCall)
(benchmark::State &state) {
    for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j) {
        result = baseObj.regularFunction(a, b);
        doNotOptimize(result);
    }
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(Dynamism, regularCall)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

// Monomorphic virtual function call
BENCHMARK_DEFINE_F(Dynamism, monomorphicCall)
(benchmark::State &state) {
    for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j) {
        result = baseObj.virtualFunction(a, b);
        doNotOptimize(result);
    }
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(Dynamism, monomorphicCall)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

// Polymorphic virtual function call
BENCHMARK_DEFINE_F(Dynamism, polymorphicCall)
(benchmark::State &state) {
    for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j) {
        result = derivedObj.virtualFunction(a, b);
        doNotOptimize(result);
    }
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(Dynamism, polymorphicCall)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);


// Monomorphic pointer virtual function call
BENCHMARK_DEFINE_F(Dynamism, monomorphicPointerCall)
(benchmark::State &state) {
    for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j) {
        result = monomorphicPtr->virtualFunction(a, b);
        doNotOptimize(result);
    }
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(Dynamism, monomorphicPointerCall)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

// Polymorphic pointer virtual function call
BENCHMARK_DEFINE_F(Dynamism, polymorphicPointerCall)
(benchmark::State &state) {
    for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j) {
        result = polymorphicPtr->virtualFunction(a, b);
        doNotOptimize(result);
    }
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(Dynamism, polymorphicPointerCall)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);



// Dynamism on hasTrait
BENCHMARK_DEFINE_F(Dynamism, hasTrait)
(benchmark::State &state) {
    ctx->loadDialect<TestBenchDialect>();
    OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
    SmallVector<Operation *> ops;
    for (int j = 0; j < state.range(0); ++j) {
        ops.push_back(b.create<OpWithRegion>(unknownLoc));
    }
    for (auto _ : state) {
    for (Operation *op : ops) {
        bool hasTrait = op->hasTrait<OpTrait::SingleBlock>();
        benchmark::DoNotOptimize(&hasTrait);
    };
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(Dynamism, hasTrait)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

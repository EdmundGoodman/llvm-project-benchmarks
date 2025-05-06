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
#include "mlir/IR/Verifier.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <unistd.h>
#include <random>

#include "benchmark/benchmark.h"

using namespace mlir;
void mlirBenchmarkInitLLVM(int argc, const char **argv);
namespace {

class SimpleConstantFolding : public benchmark::Fixture  {
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
  }

  void TearDown(::benchmark::State &state) final {
    moduleOp.release()->erase();
    ctx.reset();
  }

  void populateTestModule(int num) {
    // Seed random numbers
    std::mt19937 rng(0);
    std::uniform_int_distribution<std::mt19937::result_type> dist(1,1000); // distribution in range [1, 6]

    // Load dialects
    ctx->loadDialect<arith::ArithDialect>();
    ctx->loadDialect<TestBenchDialect>();

    OpBuilder moduleB = OpBuilder::atBlockBegin(moduleOp->getBody());
    IntegerType i32Type = moduleB.getI32Type();
    std::vector<Value> ops;

    // Initial constant op
    ops.push_back(moduleB.create<arith::ConstantOp>(
        unknownLoc,
        moduleB.getIntegerAttr(i32Type, dist(rng))));

    // Generate operations based on pattern
    for (int i = 1; i <= num + 1; i++) {
        if (i % 2 == 0) {
            // Add operation
            ops.push_back(moduleB.create<arith::AddIOp>(
                unknownLoc, ops[i - 1], ops[i - 2]));
        } else {
            // Constant operation
            ops.push_back(moduleB.create<arith::ConstantOp>(
                unknownLoc,
                moduleB.getIntegerAttr(i32Type, dist(rng))));
        }
    }

    // Final test.op operation
    moduleB.create<PassthroughOp>(unknownLoc, i32Type, ops[(num / 2) * 2]);

    if (failed(verify(moduleOp.get()))) {
        llvm::errs() << "Verifier failed " << __FILE__ << ":" << __LINE__ << "\n";
        exit(-1);
    }
  }

  std::unique_ptr<MLIRContext> ctx;
  OwningOpRef<ModuleOp> moduleOp;
  UnknownLoc unknownLoc;
};
} // namespace








struct ConstantFoldingIntegerAdditionPattern :
  public OpRewritePattern<arith::AddIOp> {
    ConstantFoldingIntegerAdditionPattern(MLIRContext *context)
      : OpRewritePattern<arith::AddIOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(arith::AddIOp op, PatternRewriter &rewriter) const override {
    // Get operands
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // Check if both operands are constants
    arith::ConstantOp lhsConstOp = lhs.getDefiningOp<arith::ConstantOp>();
    arith::ConstantOp rhsConstOp = rhs.getDefiningOp<arith::ConstantOp>();

    if (!lhsConstOp || !rhsConstOp) {
        return failure();
    }

    // Extract constant values
    auto lhsAttr = lhsConstOp.getValue().dyn_cast<IntegerAttr>();
    auto rhsAttr = rhsConstOp.getValue().dyn_cast<IntegerAttr>();
    if (!lhsAttr || !rhsAttr) {
        return failure();
    }

    // Compute the folded result
    APInt lhsValue = lhsAttr.getValue();
    APInt rhsValue = rhsAttr.getValue();
    APInt result = lhsValue + rhsValue;

    // Create a new constant operation with the folded value
    auto resultType = op.getType();
    auto foldedValue = rewriter.getIntegerAttr(resultType, result);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, foldedValue);

    return success();
  }
};








BENCHMARK_DEFINE_F(SimpleConstantFolding, folding)(benchmark::State &state) {
  ctx->disableMultithreading();
  int testSize = state.range(0);
  for (auto _ : state) {
    state.PauseTiming();
    moduleOp->getBody()->erase();
    moduleOp->getBodyRegion().push_back(new Block);
    populateTestModule(testSize);
    // moduleOp->print(llvm::outs());

    MLIRContext *context = moduleOp->getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConstantFoldingIntegerAdditionPattern>(context);

    Region& region = moduleOp.get()->getRegion(0);
    GreedyRewriteConfig config;
    config.scope = &region;
    state.ResumeTiming();


    // Fold with some optimisations manually elided
    (void)applyPatternsAndFoldGreedily(region, std::move(patterns), config);


    state.PauseTiming();
    // moduleOp->print(llvm::outs()); exit(-1);
    if (failed(verify(moduleOp.get()))) {
      llvm::errs() << "Verifier failed " << __FILE__ << ":" << __LINE__ << "\n";
      exit(-1);
    }
    state.ResumeTiming();
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(SimpleConstantFolding, folding)
    ->Ranges({{1, 10000}})
    ->Complexity(benchmark::oN);

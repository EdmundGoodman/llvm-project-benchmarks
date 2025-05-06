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
#include "mlir/Pass/Pass.h"

#include <memory>
#include <unistd.h>
#include <random>

#include "benchmark/benchmark.h"

using namespace mlir;
void mlirBenchmarkInitLLVM(int argc, const char **argv);




// Replace y = C*x with y = C/2*x + C/2*x, when C is a power of 2, otherwise do
// nothing.
struct ConstantFoldingIntegerAdditionPattern :
  public OpRewritePattern<arith::MulIOp> {
    ConstantFoldingIntegerAdditionPattern(mlir::MLIRContext *context)
      : OpRewritePattern<arith::MulIOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(arith::MulIOp op, PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!rhsDefiningOp) {
      return failure();
    }

    int64_t value = rhsDefiningOp.value();
    bool is_power_of_two = (value & (value - 1)) == 0;

    if (!is_power_of_two) {
      return failure();
    }

    arith::ConstantOp newConstant = rewriter.create<arith::ConstantOp>(
        rhsDefiningOp.getLoc(), rewriter.getIntegerAttr(rhs.getType(), value / 2));
        arith::MulIOp newMul = rewriter.create<arith::MulIOp>(op.getLoc(), lhs, newConstant);
    arith::AddIOp newAdd = rewriter.create<arith::AddIOp>(op.getLoc(), newMul, newMul);

    return success();
  }
};


// class MyPattern : public RewritePattern {
//     public:
//     //   /// This overload constructs a pattern that only matches operations with the
//     //   /// root name of `MyOp`.
//     //   MyPattern(PatternBenefit benefit, MLIRContext *context)
//     //       : RewritePattern(MyOp::getOperationName(), benefit, context) {}
//       /// This overload constructs a pattern that matches any operation type.
//       MyPattern(PatternBenefit benefit)
//           : RewritePattern(benefit, MatchAnyOpTypeTag()) {}

//       LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
//         // The `matchAndRewrite` method performs both the matching and the mutation.
//         // Note that the match must reach a successful point before IR mutation may
//         // take place.
//       }
//     };

// class ConstantFoldingIntegerAdditionPattern : public OpRewritePattern<Operation> {

//     ConstantFoldingIntegerAdditionPattern(MLIRContext *context)
//     : OpRewritePattern<Operation>(context, /*benefit=*/1) {}

//     LogicalResult matchAndRewrite(Operation operation,
//                                     PatternRewriter &rewriter) const override {
//         return successs();
//         // // Check if this is an integer addition operation
//         // auto op = dyn_cast<arith::AddIOp>(operation);
//         // if (!op) {
//         //     return failure();
//         // }

//         // // Get operands
//         // Value lhs = op.getLhs();
//         // Value rhs = op.getRhs();

//         // // Check if both operands are constants
//         // arith::ConstantOp lhsConstOp = lhs.getDefiningOp<arith::ConstantOp>();
//         // arith::ConstantOp rhsConstOp = rhs.getDefiningOp<arith::ConstantOp>();

//         // if (!lhsConstOp || !rhsConstOp) {
//         //     return failure();
//         // }

//         // // Extract constant values
//         // auto lhsAttr = lhsConstOp.getValue().dyn_cast<IntegerAttr>();
//         // auto rhsAttr = rhsConstOp.getValue().dyn_cast<IntegerAttr>();

//         // if (!lhsAttr || !rhsAttr) {
//         //     return failure();
//         // }

//         // // Compute the folded result
//         // APInt lhsValue = lhsAttr.getValue();
//         // APInt rhsValue = rhsAttr.getValue();
//         // APInt result = lhsValue + rhsValue;

//         // // Create a new constant operation with the folded value
//         // auto resultType = op.getType();
//         // auto foldedValue = rewriter.getIntegerAttr(resultType, result);
//         // rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, foldedValue);

//         // return success();
//     }
// };



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
    std::random_device dev;
    std::mt19937 rng(dev());
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
    for (int i = 1; i <= num; i++) {
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
    moduleB.create<EmptyOp>(unknownLoc);

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

BENCHMARK_DEFINE_F(SimpleConstantFolding, folding)(benchmark::State &state) {
  ctx->disableMultithreading();
  int testSize = state.range(0);
  for (auto _ : state) {
    state.PauseTiming();
    moduleOp->getBody()->erase();
    moduleOp->getBodyRegion().push_back(new Block);
    populateTestModule(testSize);
    moduleOp->print(llvm::outs());
    state.ResumeTiming();

    // for (Operation &op :
    //      llvm::make_early_inc_range(funcOp.getBody().front().getOperations())) {
    //   Attribute constValue;
    //   matchPattern(&op, m_Constant(&constValue));
    //   if (constValue) {
    //     auto it = constantsMap.insert({constValue, &op});
    //     if (!it.second) {
    //       op.replaceAllUsesWith(it.first->getSecond());
    //       op.erase();
    //     }
    //     continue;
    //   }
    //   foldResults.clear();
    //   if (succeeded(op.fold(foldResults))) {
    //     // Check to see if the operation was just updated in place.
    //     if (foldResults.empty())
    //       continue;
    //     assert(foldResults.size() == op.getNumResults());
    //     auto *dialect = op.getDialect();
    //     OpBuilder b(&op);
    //     auto loc = op.getLoc();
    //     for (auto [cst, res] : llvm::zip(foldResults, op.getResults())) {
    //       if (cst.is<Value>()) {
    //         res.replaceAllUsesWith(cst.get<Value>());
    //         if (op.use_empty())
    //           op.erase();
    //         continue;
    //       }
    //       constValue = cst.get<Attribute>();
    //       auto it = constantsMap.find(constValue);
    //       if (it != constantsMap.end()) {
    //         res.replaceAllUsesWith(it->getSecond()->getResult(0));
    //         if (op.use_empty())
    //           op.erase();
    //         continue;
    //       }

    //       auto type = res.getType();
    //       // Ask the dialect to materialize a constant operation for this value.
    //       if (auto *constOp =
    //               dialect->materializeConstant(b, constValue, type, loc)) {
    //         res.replaceAllUsesWith(constOp->getResult(0));
    //         if (op.use_empty())
    //           op.erase();
    //         auto it = constantsMap.insert({constValue, constOp});
    //         if (!it.second)
    //           llvm::report_fatal_error("Fatal");
    //       }
    //     }
    //   }
    // }
    // // By the time we are done, we may have simplified a bunch of code,
    // // leaving around dead constants. Check for them now and remove them.
    // for (auto it : constantsMap) {
    //   if (it.getSecond()->use_empty())
    //     it.getSecond()->erase();
    // }
    // state.PauseTiming();
    // // funcOp->dump();

    // if (failed(verify(moduleOp.get()))) {
    //   llvm::errs() << "Verifier failed " << __FILE__ << ":" << __LINE__ << "\n";
    //   exit(-1);
    // }
    // state.ResumeTiming();
  }

  llvm::errs() << "Benchmark hit!!\n";
  exit(1);
//   // moduleOp->dump();
//   // exit(-1);
//   int countOps = 0;
//   moduleOp->walk([&](Operation *op) { ++countOps; });
//   int expectedOps = 4; // module + func + constant + return
//   if (countOps != expectedOps) {
//     llvm::errs() << "Got " << countOps << " operation and expected "
//                  << expectedOps << "\n";
//     exit(1);
//   }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(SimpleConstantFolding, folding)
    ->Ranges({{1, 10000}})
    ->Complexity(benchmark::oN);

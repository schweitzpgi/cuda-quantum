/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_WIRESETTOMEM
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "wireset-to-mem"

using namespace mlir;

namespace {
struct WireSetAnalysis {
  explicit WireSetAnalysis(ModuleOp m) : module(m) {}

  void setProcessed(func::FuncOp fun) {
    processedFuncs.insert(fun.getOperation());
  }
  void setProcessed(func::ReturnOp ret) {
    processedReturns.insert(ret.getOperation());
  }

  bool wasProcessed(func::FuncOp fun) {
    if (!funSets.count(fun.getOperation()))
      return true;
    return processedFuncs.count(fun.getOperation());
  }
  bool wasProcessed(func::ReturnOp ret) {
    return processedReturns.count(ret.getOperation());
  }

  void addWireSetToFunction(StringRef name, Operation *fun) {
    funSets[fun].insert(name.data());
  }

  quake::AllocaOp retrieveAlloca(Operation *fun, StringRef setName) {
    return cast<quake::AllocaOp>(allocaMap[fun][setName.data()]);
  }

  SmallPtrSetImpl<const char *> &getWireSetNames(func::FuncOp fun) {
    auto iter = funSets.find(fun);
    assert(iter != funSets.end());
    return iter->second;
  }

  quake::WireSetOp getWireSetByName(StringRef name) {
    auto result = module.lookupSymbol<quake::WireSetOp>(name);
    assert(result);
    return result;
  }

  ModuleOp getModule() { return module; }

  void addAllocationToFunc(const char *name, Operation *alloc, Operation *fun) {
    allocaMap[fun][name] = alloc;
  }

  SmallVector<Operation *> getAllocsForFunc(Operation *fun) {
    SmallVector<Operation *> results;
    for (auto [key, val] : allocaMap[fun])
      results.push_back(val);
    return results;
  }

private:
  ModuleOp module;
  // Map from FuncOp to sets of names of WireSet ops referenced by function.
  DenseMap<Operation *, SmallPtrSet<const char *, 4>> funSets;
  // Map from FuncOp to a map from WireSet name to quake::AllocaOp.
  DenseMap<Operation *, DenseMap<const char *, Operation *>> allocaMap;
  // Set of func::FuncOp already processed.
  SmallPtrSet<Operation *, 4> processedFuncs;
  // Set of func::ReturnOp already processed.
  SmallPtrSet<Operation *, 4> processedReturns;
};

struct FuncFuncPattern : public OpRewritePattern<func::FuncOp> {
  explicit FuncFuncPattern(MLIRContext *ctx, WireSetAnalysis &analysis)
      : OpRewritePattern(ctx), analysis(analysis) {}

  LogicalResult matchAndRewrite(func::FuncOp fun,
                                PatternRewriter &rewriter) const override {
    if (analysis.wasProcessed(fun))
      return failure();

    // Go through the set of wire-set names and allocate each one.
    auto *ctx = rewriter.getContext();
    rewriter.setInsertionPointToStart(&fun.getRegion().front());
    auto loc = fun.getLoc();
    for (auto *name : analysis.getWireSetNames(fun)) {
      auto wireSet = analysis.getWireSetByName(name);
      auto veqTy = quake::VeqType::get(ctx, wireSet.getCardinality());
      auto alloc = rewriter.create<quake::AllocaOp>(loc, veqTy);
      analysis.addAllocationToFunc(name, alloc, fun);
    }
    analysis.setProcessed(fun);
    return success();
  }

private:
  WireSetAnalysis &analysis;
};

struct FuncReturnPattern : public OpRewritePattern<func::ReturnOp> {
  explicit FuncReturnPattern(MLIRContext *ctx, WireSetAnalysis &analysis)
      : OpRewritePattern(ctx), analysis(analysis) {}

  LogicalResult matchAndRewrite(func::ReturnOp ret,
                                PatternRewriter &rewriter) const override {
    if (analysis.wasProcessed(ret))
      return failure();
    auto fun = ret->getParentOfType<func::FuncOp>();
    if (!analysis.wasProcessed(fun))
      return failure();

    // Go through the set of wire-set allocations and deallocate each one.
    auto loc = ret.getLoc();
    for (auto *op : analysis.getAllocsForFunc(fun))
      rewriter.create<quake::DeallocOp>(loc, op->getResult(0));

    analysis.setProcessed(ret);
    return success();
  }

private:
  WireSetAnalysis &analysis;
};

struct BorrowWirePattern : public OpRewritePattern<quake::BorrowWireOp> {
  explicit BorrowWirePattern(MLIRContext *ctx, WireSetAnalysis &analysis)
      : OpRewritePattern(ctx), analysis(analysis) {}

  LogicalResult matchAndRewrite(quake::BorrowWireOp borrow,
                                PatternRewriter &rewriter) const override {
    // Convert borrow to an extract_ref + unwrap.
    auto fun = borrow->getParentOfType<func::FuncOp>();
    assert(fun && "borrow_wire must be in a FuncOp");
    auto alloc = analysis.retrieveAlloca(fun, borrow.getSetName());
    auto *ctx = rewriter.getContext();
    Value ref = rewriter.create<quake::ExtractRefOp>(borrow.getLoc(), alloc,
                                                     borrow.getIdentity());
    auto wireTy = quake::WireType::get(ctx);
    rewriter.replaceOpWithNewOp<quake::UnwrapOp>(borrow, wireTy, ref);
    return success();
  }

private:
  WireSetAnalysis &analysis;
};

struct ReturnWirePattern : public OpRewritePattern<quake::ReturnWireOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::ReturnWireOp retWire,
                                PatternRewriter &rewriter) const override {
    // Convert ReturnWire to a Sink.
    rewriter.replaceOpWithNewOp<quake::SinkOp>(retWire, retWire.getTarget());
    return success();
  }
};

struct WireSetToMemPass
    : public cudaq::opt::impl::WireSetToMemBase<WireSetToMemPass> {
  using WireSetToMemBase::WireSetToMemBase;

  static bool hasWireSet(ModuleOp mod) {
    for (Operation &op : *mod.getBody())
      if (isa<quake::WireSetOp>(op))
        return true;
    return false;
  }

  static WireSetAnalysis getWireSetAnalysis(ModuleOp mod) {
    WireSetAnalysis result(mod);
    mod.walk([&](quake::BorrowWireOp borrow) {
      auto fun = borrow->getParentOfType<func::FuncOp>();
      result.addWireSetToFunction(borrow.getSetName(), fun);
    });
    return result;
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    ModuleOp mod = getOperation();

    if (!hasWireSet(mod))
      return;
    auto analysis = getWireSetAnalysis(mod);

    RewritePatternSet patterns(ctx);
    patterns.insert<ReturnWirePattern>(ctx);
    patterns.insert<BorrowWirePattern, FuncFuncPattern, FuncReturnPattern>(
        ctx, analysis);
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

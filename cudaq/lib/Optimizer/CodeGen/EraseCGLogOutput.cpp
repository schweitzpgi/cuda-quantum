/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/CompilerNames.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_ERASECOMPILERGENERATEDLOGOUTPUT
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "erase-implicit-output"

using namespace mlir;

namespace {

class EraseCompilerGeneratedLogOutputPass
    : public cudaq::opt::impl::EraseCompilerGeneratedLogOutputBase<
          EraseCompilerGeneratedLogOutputPass> {
public:
  using EraseCompilerGeneratedLogOutputBase::
      EraseCompilerGeneratedLogOutputBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    for (Block &block : funcOp.getBody())
      for (Operation &op : llvm::make_early_inc_range(block))
        if (auto logOut = dyn_cast<cudaq::cc::LogOutputOp>(op))
          if (logOut->hasAttr(cudaq::runtime::compilerGeneratedOutput))
            logOut->erase();
  }
};
} // namespace

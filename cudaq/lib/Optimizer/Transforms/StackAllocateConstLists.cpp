/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_STACKALLOCATECONSTLISTS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "stack-allocate-const-lists"

using namespace mlir;

/**
   \file

   See the `StackAllocateConstLists` description in `Passes.td` for the
   motivation.

   This pass looks for the "free the old buffer (if any), then malloc a new
   one" idiom the Python AST bridge emits for a dynamically-sized list (see
   `ast_bridge.py`'s `__cudaq__check_and_reallocate` / `visit_ListComp`), in
   either of the two shapes it may appear in by the time this pass runs:

     1. Still as calls to the named intrinsics:
          %new = call @__cudaq__check_and_reallocate(%old, %bytes)
        (and, independently, a call to @__cudaq__check_and_free(%ptr) for a
        buffer's own trailing free.)

     2. Already inlined into its constituent operations (this is what
        actually happens once a kernel has been compiled - and hence
        inlined - before argument synthesis ever runs):
          %i0 = cc.cast %old : (!cc.ptr<i8>) -> i64
          %nz = arith.cmpi ne, %i0, %c0 : i64
          cc.if(%nz) {
            func.call @free(%old) : (!cc.ptr<i8>) -> ()
          }
          %new = call @malloc(%bytes) : (i64) -> !cc.ptr<i8>

   When %bytes is a compile-time constant (which it was not when the bridge
   built this, but may become after `argument-synthesis` substitutes a
   concrete value for a kernel parameter), %new's buffer is a fixed-size,
   not a dynamically-sized, local, so it is rewritten to a plain `cc.alloca`
   - exactly the code the bridge would have produced had it known the size
   was constant from the start.
*/

namespace {

/// Strip a chain of `cc.cast` ops down to the underlying value.
static Value stripCasts(Value v) {
  while (auto cast = v.getDefiningOp<cudaq::cc::CastOp>())
    v = cast.getValue();
  return v;
}

static bool isProvablyNullPointer(Value v) {
  v = stripCasts(v);
  APInt cst;
  return matchPattern(v, m_ConstantInt(&cst)) && cst.isZero();
}

/// Matches a `cc.if` immediately preceding `beforeOp` in the same block of
/// the shape:
///   %i = cc.cast %ptr : (!cc.ptr<i8>) -> i64
///   %nz = arith.cmpi ne, %i, %c0 : i64
///   cc.if(%nz) {
///     func.call @free(%ptr) : (!cc.ptr<i8>) -> ()
///   } [else {}]
/// Returns the freed pointer (`%ptr`) and the matched `cc.if`, or a null
/// value/op if the immediately preceding operation is not this idiom.
static std::pair<Value, cudaq::cc::IfOp>
matchLeadingCheckAndFree(Operation *beforeOp) {
  auto *prev = beforeOp->getPrevNode();
  auto ifOp = dyn_cast_or_null<cudaq::cc::IfOp>(prev);
  if (!ifOp || ifOp.getNumResults() != 0)
    return {};
  if (ifOp.hasElse() && !ifOp.getElseEntryBlock()->without_terminator().empty())
    return {};
  Block *thenBlock = ifOp.getThenEntryBlock();
  auto ops = thenBlock->without_terminator();
  if (ops.empty() || std::next(ops.begin()) != ops.end())
    return {};
  auto call = dyn_cast<func::CallOp>(&*ops.begin());
  if (!call || call.getCallee() != "free" || call.getNumOperands() != 1)
    return {};
  Value freedPtr = call.getOperand(0);

  Value cond = ifOp.getCondition();
  auto cmp = cond.getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::ne)
    return {};
  APInt zero;
  Value lhs = cmp.getLhs(), rhs = cmp.getRhs();
  Value intVal;
  if (matchPattern(rhs, m_ConstantInt(&zero)) && zero.isZero())
    intVal = lhs;
  else if (matchPattern(lhs, m_ConstantInt(&zero)) && zero.isZero())
    intVal = rhs;
  else
    return {};
  auto castToInt = intVal.getDefiningOp<cudaq::cc::CastOp>();
  if (!castToInt || stripCasts(castToInt.getValue()) != stripCasts(freedPtr))
    return {};
  return {freedPtr, ifOp};
}

/// Erases every "free the buffer" use of `ptr` (matched via SSA identity
/// after stripping casts) found anywhere in `func`: a bare
/// `call @__cudaq__check_and_free(ptr)`, or the inlined
/// `cc.cast`/`arith.cmpi`/`cc.if{call @free}` idiom above.
static void eraseAllFreesOf(Value ptr, func::FuncOp func) {
  Value target = stripCasts(ptr);
  SmallVector<Operation *> toErase;
  func.walk([&](func::CallOp call) {
    if (call.getCallee() == "__cudaq__check_and_free" &&
        call.getNumOperands() == 1 && stripCasts(call.getOperand(0)) == target)
      toErase.push_back(call);
  });
  func.walk([&](cudaq::cc::IfOp ifOp) {
    // Reuse the leading-idiom matcher by pretending we are matching just
    // before this cc.if's own next node - i.e. treat the cc.if itself as
    // "beforeOp" is wrong; instead re-derive the same shape directly here
    // rooted at ifOp.
    if (ifOp.getNumResults() != 0)
      return;
    if (ifOp.hasElse() &&
        !ifOp.getElseEntryBlock()->without_terminator().empty())
      return;
    Block *thenBlock = ifOp.getThenEntryBlock();
    auto ops = thenBlock->without_terminator();
    if (ops.empty() || std::next(ops.begin()) != ops.end())
      return;
    auto call = dyn_cast<func::CallOp>(&*ops.begin());
    if (!call || call.getCallee() != "free" || call.getNumOperands() != 1)
      return;
    if (stripCasts(call.getOperand(0)) != target)
      return;
    Value cond = ifOp.getCondition();
    auto cmp = cond.getDefiningOp<arith::CmpIOp>();
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::ne)
      return;
    toErase.push_back(ifOp);
    if (auto cast = cmp.getLhs().getDefiningOp<cudaq::cc::CastOp>())
      if (stripCasts(cast.getValue()) == target)
        toErase.push_back(cast);
    if (auto cast = cmp.getRhs().getDefiningOp<cudaq::cc::CastOp>())
      if (stripCasts(cast.getValue()) == target)
        toErase.push_back(cast);
    toErase.push_back(cmp);
  });
  for (Operation *op : toErase)
    if (op->use_empty())
      op->erase();
}

/// If `ptr`'s only remaining uses are `cc.cast`s, erase them once `ptr`
/// itself is dead. Used to clean up the ptrtoint casts a removed
/// check-and-free idiom leaves dangling.
static void eraseDeadCastsOf(Value v) {
  SmallVector<Operation *> users(v.getUsers().begin(), v.getUsers().end());
  for (Operation *user : users)
    if (auto cast = dyn_cast<cudaq::cc::CastOp>(user))
      if (cast->use_empty())
        cast->erase();
}

/// Recovers the element type of the array a `malloc`ed `!cc.ptr<i8>` was
/// meant to back, from a `cc.cast` of `ptr` to `!cc.ptr<!cc.array<T x ?>>>`
/// (or `!cc.ptr<T>`) among its uses.
static Type findElementType(Value ptr) {
  for (Operation *user : ptr.getUsers()) {
    auto cast = dyn_cast<cudaq::cc::CastOp>(user);
    if (!cast)
      continue;
    auto ptrTy = dyn_cast<cudaq::cc::PointerType>(cast.getType());
    if (!ptrTy)
      continue;
    if (auto arrTy = dyn_cast<cudaq::cc::ArrayType>(ptrTy.getElementType()))
      return arrTy.getElementType();
    // A direct `!cc.ptr<i8>` cast (the pointer's own storage type) tells us
    // nothing about the buffer's real element type; skip it and keep
    // looking at other uses.
    if (ptrTy.getElementType().isInteger(8))
      continue;
    return ptrTy.getElementType();
  }
  return {};
}

static bool typeHasKnownSize(Type ty) {
  return isa<IntegerType, FloatType, ComplexType, cudaq::cc::StructType>(ty);
}

class StackAllocateConstListsPass
    : public cudaq::opt::impl::StackAllocateConstListsBase<
          StackAllocateConstListsPass> {
public:
  using StackAllocateConstListsBase::StackAllocateConstListsBase;

  void runOnOperation() override {
    auto func = getOperation();
    auto module = func->getParentOfType<ModuleOp>();
    DataLayout dataLayout(module);
    SmallPtrSet<Operation *, 8> convertedResults;

    // Two candidate call shapes: the still-outlined intrinsic, and a bare
    // `malloc` (the shape it takes once inlined).
    SmallVector<func::CallOp> candidates;
    func.walk([&](func::CallOp call) {
      if (call.getCallee() == "__cudaq__check_and_reallocate" ||
          call.getCallee() == "malloc")
        candidates.push_back(call);
    });

    for (func::CallOp call : candidates) {
      bool isReallocCall = call.getCallee() == "__cudaq__check_and_reallocate";
      Value bytesVal = call.getOperand(isReallocCall ? 1 : 0);
      Value oldPtr;
      cudaq::cc::IfOp leadingIf;
      if (isReallocCall) {
        oldPtr = call.getOperand(0);
      } else {
        std::tie(oldPtr, leadingIf) = matchLeadingCheckAndFree(call);
      }

      APInt bytesConst;
      if (!matchPattern(bytesVal, m_ConstantInt(&bytesConst)))
        continue;

      Value result = call.getResult(0);
      Type elemTy = findElementType(result);
      if (!elemTy || !typeHasKnownSize(elemTy))
        continue;
      uint64_t elemSize = dataLayout.getTypeSize(elemTy);
      uint64_t totalBytes = bytesConst.getZExtValue();
      if (elemSize == 0 || totalBytes % elemSize != 0)
        continue;
      uint64_t elemCount = totalBytes / elemSize;

      OpBuilder builder(call);
      Location loc = call.getLoc();
      Value countVal =
          arith::ConstantOp::create(builder, loc, builder.getI64Type(),
                                    builder.getI64IntegerAttr(elemCount));
      auto alloca = cudaq::cc::AllocaOp::create(builder, loc, elemTy, countVal);
      Value newPtr = cudaq::cc::CastOp::create(
          builder, loc, cudaq::cc::PointerType::get(builder.getI8Type()),
          alloca.getResult());
      result.replaceAllUsesWith(newPtr);
      convertedResults.insert(alloca);

      // The buffer this call/idiom just replaced is now dead code from a
      // memory-management point of view: free it only if doing so is
      // provably safe (a literal null, or a buffer this same pass has
      // already turned into a `cc.alloca`); otherwise leave whatever freed
      // it alone rather than risk a leak.
      bool oldIsSafeToDrop =
          oldPtr &&
          (isProvablyNullPointer(oldPtr) ||
           convertedResults.contains(stripCasts(oldPtr).getDefiningOp()));
      if (isReallocCall) {
        if (!oldIsSafeToDrop)
          func::CallOp::create(builder, loc, TypeRange{},
                               "__cudaq__check_and_free", ValueRange{oldPtr});
      } else if (leadingIf && oldIsSafeToDrop) {
        Value freedPtr = stripCasts(oldPtr);
        auto cmp = leadingIf.getCondition().getDefiningOp<arith::CmpIOp>();
        cudaq::cc::CastOp intCast;
        if (cmp) {
          intCast = cmp.getLhs().getDefiningOp<cudaq::cc::CastOp>();
          if (!intCast)
            intCast = cmp.getRhs().getDefiningOp<cudaq::cc::CastOp>();
        }
        leadingIf.erase();
        if (cmp && cmp->use_empty())
          cmp.erase();
        if (intCast && intCast->use_empty())
          intCast.erase();
        eraseDeadCastsOf(freedPtr);
      }

      call.erase();

      // A `free` of *this* buffer's own result is always wrong now (it
      // would call `free` on a stack pointer), regardless of whether it
      // was provably safe to drop the *previous* buffer above.
      eraseAllFreesOf(newPtr, func);
      eraseDeadCastsOf(newPtr);
    }
  }
};
} // namespace

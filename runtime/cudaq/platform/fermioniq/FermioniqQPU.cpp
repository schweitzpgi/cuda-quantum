/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "FermioniqBaseQPU.h"
// #include "common/BaseRemoteRESTQPU.h"

using namespace mlir;

namespace cudaq {
std::string get_quake_by_name(const std::string &);
} // namespace cudaq

namespace {

/// @brief The `FermioniqRestQPU` is a subtype of QPU that enables the
/// execution of CUDA-Q kernels on the Fermioniq simulator via a REST Client.
class FermioniqRestQPU : public cudaq::FermioniqBaseQPU {
protected:
  std::tuple<ModuleOp, MLIRContext *, void *>
  extractQuakeCodeAndContext(const std::string &kernelName,
                             void *data) override {

    cudaq::info("extract quake code\n");

    auto contextPtr = cudaq::initializeMLIR();
    MLIRContext &context = *contextPtr.get();

    // Get the quake representation of the kernel
    auto quakeCode = cudaq::get_quake_by_name(kernelName);
    auto m_module = parseSourceString<ModuleOp>(quakeCode, &context);
    if (!m_module)
      throw std::runtime_error("module cannot be parsed");

    return std::make_tuple(m_module.release(), contextPtr.release(), data);
  }

  void cleanupContext(MLIRContext *context) override { delete context; }

public:
  /// @brief The constructor
  FermioniqRestQPU() : FermioniqBaseQPU() {}

  FermioniqRestQPU(FermioniqRestQPU &&) = delete;
  virtual ~FermioniqRestQPU() = default;
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, FermioniqRestQPU, fermioniq)

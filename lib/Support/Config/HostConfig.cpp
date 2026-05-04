/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Support/HostConfig.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Host.h"

std::string cudaq::config::getProcessTriple() {
  return llvm::sys::getProcessTriple();
}

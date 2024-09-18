/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: split-file %s %t && \
// RUN: nvq++ -target quantinuum -emulate -fno-set-target-backend -c %t/emulib.cpp -o %t/emulibx.o && \
// RUN: nvq++ -target quantinuum -emulate -c %t/emuuser.cpp -o %t/emuuserx.o && \
// RUN: nvq++ -target quantinuum -emulate %t/emulibx.o %t/emuuserx.o -o %t/emux.a.out && \
// RUN: %t/emux.a.out | FileCheck %s
// clang-format on

// Catch-22. We need to use argument synthesis, but if I enable that in the
// quantum_platform.cpp code, there are a bunch of failures.

//--- emulib.h

#include "cudaq.h"

__qpu__ void dunkadee(cudaq::qvector<> &q);

//--- emulib.cpp

#include "emulib.h"
#include <iostream>

__qpu__ void dunkadee(cudaq::qvector<> &q) { x(q[0]); }

//--- emuuser.cpp

#include "emulib.h"
#include <iostream>

__qpu__ void
userKernel(const cudaq::qkernel_ref<void(cudaq::qvector<> &)> &init) {
  cudaq::qvector q(2);
  init(q);
}

int main() {
  cudaq::sample(10, userKernel,
                cudaq::qkernel_ref<void(cudaq::qvector<> &)>{dunkadee});
  std::cout << "Hello, World!\n";
  return 0;
}

// CHECK: Hello, World

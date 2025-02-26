/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %cpp_std %s | cudaq-opt | cudaq-translate --convert-to=qir | FileCheck %s
// clang-format on

#include <cudaq.h>

struct SantaKraus : public cudaq::kraus_channel {
  constexpr static std::size_t num_parameters = 1;
  constexpr static std::size_t num_targets = 1;
  static std::size_t get_key() { return (std::size_t)&get_key; }
  SantaKraus(double trouble, cudaq::qubit &cubit) {}
};

template <typename T>
struct bell_error_vec {
  void operator()(double prob) __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    cudaq::apply_noise<T>(prob, q);
  }
};

int main() {
  cudaq::noise_model noise;
  auto counts =
      cudaq::sample({.noise = noise}, bell_error_vec<SantaKraus>{}, 0.5);
  counts.dump();
  return 0;
}

// clang-format off
// CHECK-LABEL: define void @__nvqpp__mlirgen__bell_error_vecI10SantaKrausE(double 
// CHECK-SAME:    %[[VAL_0:.*]]) local_unnamed_addr {
// CHECK:         %[[VAL_1:.*]] = tail call %[[VAL_2:.*]]* @__quantum__rt__qubit_allocate_array(i64 2)
// CHECK:         %[[VAL_3:.*]] = alloca double, align 8
// CHECK:         store double %[[VAL_0]], double* %[[VAL_3]], align 8
// CHECK:         %[[VAL_4:.*]] = tail call %[[VAL_5:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_2]]* %[[VAL_1]], i64 0)
// CHECK:         %[[VAL_6:.*]] = load %[[VAL_5]]*, %[[VAL_5]]** %[[VAL_4]], align 8
// CHECK:         tail call void @__quantum__qis__h(%[[VAL_5]]* %[[VAL_6]])
// CHECK:         %[[VAL_7:.*]] = tail call %[[VAL_5]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_2]]* %[[VAL_1]], i64 1)
// CHECK:         %[[VAL_8:.*]] = bitcast %[[VAL_5]]** %[[VAL_7]] to i8**
// CHECK:         %[[VAL_9:.*]] = load i8*, i8** %[[VAL_8]], align 8
// CHECK:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (%[[VAL_2]]*, %[[VAL_5]]*)* @__quantum__qis__x__ctl to i8*), %[[VAL_5]]* %[[VAL_6]], i8* %[[VAL_9]])
// CHECK:         %[[VAL_10:.*]] = alloca { i8*, i8*, i8* }, align 8
// CHECK:         %[[VAL_11:.*]] = bitcast { i8*, i8*, i8* }* %[[VAL_10]] to %[[VAL_2]]*
// CHECK:         call void @__quantum__qis__convert_array_to_qvector(%[[VAL_2]]* nonnull %[[VAL_11]], %[[VAL_2]]* %[[VAL_1]])
// CHECK:         call void @_ZN5cudaq11apply_noiseI10SantaKrausJRdRNS_7qvectorILm2EEEEEEvDpOT0_(double* nonnull %[[VAL_3]], %[[VAL_2]]* nonnull %[[VAL_11]])
// CHECK:         call void @__quantum__qis__qvector_destructor(%[[VAL_2]]* nonnull %[[VAL_11]])
// CHECK:         call void @__quantum__rt__qubit_release_array(%[[VAL_2]]* %[[VAL_1]])
// CHECK:         ret void
// CHECK:       }

// CHECK: declare void @_ZN5cudaq11apply_noiseI10SantaKrausJRdRNS_7qvectorILm2EEEEEEvDpOT0_(double*, %[[VAL_12:.*]]*) local_unnamed_addr

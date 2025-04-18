/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/ExecutionContext.h"
#include "nvqir/Gates.h"
#include <cmath>

extern "C" {
extern bool verbose;

// Runtime Init / Finalize
void __quantum__rt__initialize(int argc, int8_t **argv);
void __quantum__rt__finalize();

int info(const char *format, ...);
int error(const char *format, ...);

class Array;
class Qubit;
using Result = bool;

using TuplePtr = int8_t *;

// Quantum instruction set - quantum intrinsic operations
void __quantum__qis__swap(Qubit *src, Qubit *tgt);
void __quantum__qis__swap__ctl(Array *ctrls, Qubit *q, Qubit *r);
void __quantum__qis__cnot(Qubit *src, Qubit *tgt);
void __quantum__qis__cz(Qubit *src, Qubit *tgt);
void __quantum__qis__cphase(double x, Qubit *src, Qubit *tgt);
void __quantum__qis__h(Qubit *q);
void __quantum__qis__h__ctl(Array *ctls, Qubit *q);

void __quantum__qis__s(Qubit *q);
void __quantum__qis__s__ctl(Array *ctls, Qubit *q);

void __quantum__qis__sdg(Qubit *q);
void __quantum__qis__t(Qubit *q);
void __quantum__qis__t__ctl(Array *ctls, Qubit *q);
void __quantum__qis__tdg(Qubit *q);
void __quantum__qis__reset(Qubit *q);
void __quantum__qis__x(Qubit *q);
void __quantum__qis__x__ctl(Array *ctls, Qubit *q);
void __quantum__qis__y(Qubit *q);
void __quantum__qis__y__ctl(Array *ctls, Qubit *q);
void __quantum__qis__z(Qubit *q);
void __quantum__qis__z__ctl(Array *ctls, Qubit *q);
void __quantum__qis__rx(double x, Qubit *q);
void __quantum__qis__rx__ctl(double x, Array *ctrls, Qubit *q);
void __quantum__qis__ry(double x, Qubit *q);
void __quantum__qis__ry__ctl(double x, Array *ctrls, Qubit *q);
void __quantum__qis__rz(double x, Qubit *q);
void __quantum__qis__rz__ctl(double x, Array *ctrls, Qubit *q);
void __quantum__qis__u3(double theta, double phi, double lambda, Qubit *q);
Result *__quantum__qis__mz(Qubit *q);
Result *__quantum__qis__measure__body(Array *basis, Array *qubits);
Result *__quantum__rt__result_get_one();
Result *__quantum__rt__result_get_zero();
void __quantum__qis__exp__body(Array *paulis, double angle, Array *qubits);
// Utility function used by MLIRGen to map Qubit*... controls to Array*
void invokeWithControlQubits(const std::size_t nControls,
                             void (*QISFunction)(Array *, Qubit *), ...);

void __quantum__qis__apply__general_qubit_array(Array *data, Array *qubits);
void __quantum__qis__apply__general(Array *data, int64_t n_qubits, ...);

void __quantum__qis__apply_kraus_channel_generalized(
    std::int64_t dataKind, std::int64_t krausChannelKey, std::size_t numSpans,
    std::size_t numParams, std::size_t numTargets, ...);
void __quantum__qis__apply_kraus_channel_double(std::int64_t krausChannelKey,
                                                double *params,
                                                std::size_t numParams,
                                                Array *qubits);
// Qubit array allocation / deallocation
Array *__quantum__rt__qubit_allocate_array(uint64_t idx);
Array *__quantum__rt__qubit_allocate_array_with_state_complex64(
    std::uint64_t size, std::complex<double> *data);
Array *__quantum__rt__qubit_allocate_array_with_state_complex32(
    uint64_t size, std::complex<float> *data);
Array *__quantum__rt__qubit_allocate_array_with_state_ptr(
    cudaq::SimulationState *state);
void __quantum__rt__qubit_release_array(Array *q);
void __quantum__rt__qubit_release(Qubit *q);
Qubit *__quantum__rt__qubit_allocate();

// I think all variational / nisq tasks can be
// expressed with these functions
// TuplePtr should be a struct that contains a name and
// a void* pointer to result data.
void __quantum__rt__setExecutionContext(cudaq::ExecutionContext *context);
void __quantum__rt__resetExecutionContext();

// Array utility functions
Array *__quantum__rt__array_create_1d(int32_t itemSizeInBytes,
                                      int64_t count_items);
void __quantum__rt__array_release(Array *);
int64_t __quantum__rt__array_get_size_1d(Array *array);
int8_t *__quantum__rt__array_get_element_ptr_1d(Array *q, uint64_t idx);
Array *__quantum__rt__array_slice(Array *array, int32_t dim,
                                  int64_t range_start, int64_t range_step,
                                  int64_t range_end);
Array *__quantum__rt__array_slice_1d(Array *array, int64_t range_start,
                                     int64_t range_step, int64_t range_end);
}

CUDAQ_TEST(NVQIRTester, checkSimple) {
  __quantum__rt__initialize(0, nullptr);
  auto qubits = __quantum__rt__qubit_allocate_array(2);
  Qubit *q1 = *reinterpret_cast<Qubit **>(
      __quantum__rt__array_get_element_ptr_1d(qubits, 0));
  Qubit *q2 = *reinterpret_cast<Qubit **>(
      __quantum__rt__array_get_element_ptr_1d(qubits, 1));

  __quantum__qis__h(q1);
  __quantum__qis__cnot(q1, q2);
  auto r = __quantum__qis__mz(q1);
  auto s = __quantum__qis__mz(q2);
  EXPECT_EQ(*r, *s);

  __quantum__rt__qubit_release_array(qubits);
  __quantum__rt__finalize();
}

// Stim does not support many of the gates used in these tests.
#ifndef CUDAQ_BACKEND_STIM

CUDAQ_TEST(NVQIRTester, checkQuantumIntrinsics) {
  __quantum__rt__initialize(0, nullptr);
  auto qubits = __quantum__rt__qubit_allocate_array(3);
  Qubit *src = *reinterpret_cast<Qubit **>(
      __quantum__rt__array_get_element_ptr_1d(qubits, 0));
  Qubit *tgt = *reinterpret_cast<Qubit **>(
      __quantum__rt__array_get_element_ptr_1d(qubits, 1));

  auto q = src;
  auto ctls = __quantum__rt__array_slice_1d(qubits, 2, 1, 2);
  __quantum__qis__swap(src, tgt);
  __quantum__qis__swap__ctl(ctls, src, tgt);
  __quantum__qis__cnot(src, tgt);
  __quantum__qis__cphase(2.2, src, tgt);
  __quantum__qis__h(q);
  __quantum__qis__h__ctl(ctls, q);

  __quantum__qis__s(q);
  __quantum__qis__s__ctl(ctls, q);

  __quantum__qis__sdg(q);
  __quantum__qis__t(q);
  __quantum__qis__t__ctl(ctls, q);
  __quantum__qis__tdg(q);
  __quantum__qis__reset(q);
  __quantum__qis__x(q);
  __quantum__qis__x__ctl(ctls, q);
  __quantum__qis__y(q);
  __quantum__qis__y__ctl(ctls, q);
  __quantum__qis__z(q);
  __quantum__qis__z__ctl(ctls, q);
  __quantum__qis__rx(2.2, q);
  __quantum__qis__ry(2.2, q);
  __quantum__qis__rz(2.2, q);
  __quantum__qis__u3(1.1, 2.2, 3.3, q);
  __quantum__rt__qubit_release_array(qubits);
  __quantum__rt__finalize();
}

CUDAQ_TEST(NVQIRTester, checkReset) {
  __quantum__rt__initialize(0, nullptr);
  auto qubits = __quantum__rt__qubit_allocate_array(2);
  Qubit *q0 = *reinterpret_cast<Qubit **>(
      __quantum__rt__array_get_element_ptr_1d(qubits, 0));
  Qubit *q1 = *reinterpret_cast<Qubit **>(
      __quantum__rt__array_get_element_ptr_1d(qubits, 1));

#if defined CUDAQ_BACKEND_TENSORNET
  // Tensornet backends don't have a qubit count limit, just check that it can
  // perform qubit reset in a loop.
  constexpr int N_ITERS = 3;
#else
  constexpr int N_ITERS = 100;
#endif
  // Make sure that the state vector doesn't grow with each additional reset
  for (int i = 0; i < N_ITERS; i++) {
    __quantum__qis__reset(q0);
    __quantum__qis__reset(q1);
    __quantum__qis__x(q1);
    __quantum__qis__swap(q0, q1);
    assert(*__quantum__qis__mz(q0) == 1);
  }
  __quantum__rt__qubit_release_array(qubits);
  __quantum__rt__finalize();
}

// SWAP with a single ctrl qubit in 0 state.
CUDAQ_TEST(NVQIRTester, checkSWAP) {
  // Simple SWAP.
  {
    __quantum__rt__initialize(0, nullptr);
    auto qubits = __quantum__rt__qubit_allocate_array(3);
    Qubit *q0 = *reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(qubits, 0));
    Qubit *q1 = *reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(qubits, 1));

    // Place qubit 0 in the 1-state.
    __quantum__qis__x(q0);

    // Swap qubits 0 and 1.
    __quantum__qis__swap(q0, q1);

    assert(*__quantum__qis__mz(q0) == 0);
    assert(*__quantum__qis__mz(q1) == 1);

    __quantum__rt__qubit_release_array(qubits);
    __quantum__rt__finalize();
  }

  {
    __quantum__rt__initialize(0, nullptr);
    auto ctrls = __quantum__rt__qubit_allocate_array(1);
    auto qubits = __quantum__rt__qubit_allocate_array(2);
    // Qubit *ctrl = *reinterpret_cast<Qubit **>(
    //     __quantum__rt__array_get_element_ptr_1d(ctrls, 0));
    Qubit *q0 = *reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(qubits, 0));
    Qubit *q1 = *reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(qubits, 1));

    // Place qubit 0 in the 1-state.
    __quantum__qis__x(q1);

    // Swap qubits 0 and 1 based on state of the single ctrl.
    __quantum__qis__swap__ctl(ctrls, q0, q1);

    assert(*__quantum__qis__mz(q0) == 0);
    assert(*__quantum__qis__mz(q1) == 1);

    __quantum__rt__qubit_release_array(ctrls);
    __quantum__rt__qubit_release_array(qubits);
    __quantum__rt__finalize();
  }

  // SWAP with three ctrl qubits in 1 state.
  {
    __quantum__rt__initialize(0, nullptr);
    auto ctrls = __quantum__rt__qubit_allocate_array(3);
    Qubit *ctrl0 = *reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(ctrls, 0));
    Qubit *ctrl1 = *reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(ctrls, 1));
    Qubit *ctrl2 = *reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(ctrls, 2));
    auto qubits = __quantum__rt__qubit_allocate_array(2);
    Qubit *q0 = *reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(qubits, 0));
    Qubit *q1 = *reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(qubits, 1));

    // Place controls in 1 state.
    __quantum__qis__x(ctrl0);
    __quantum__qis__x(ctrl1);
    __quantum__qis__x(ctrl2);

    // Place qubit 1 in the 1-state.
    __quantum__qis__x(q1);

    // Swap qubits 0 and 1 based on state of the single ctrl.
    __quantum__qis__swap__ctl(ctrls, q0, q1);

    assert(*__quantum__qis__mz(q0) == 1);
    assert(*__quantum__qis__mz(q1) == 0);

    __quantum__rt__qubit_release_array(ctrls);
    __quantum__rt__qubit_release_array(qubits);
    __quantum__rt__finalize();
  }
}

CUDAQ_TEST(NVQIRTester, checkQubitReset) {
  // Initialize two qubits in the 0-state.
  __quantum__rt__initialize(0, nullptr);
  auto qubits = __quantum__rt__qubit_allocate_array(2);
  Qubit *q1 = *reinterpret_cast<Qubit **>(
      __quantum__rt__array_get_element_ptr_1d(qubits, 0));
  Qubit *q2 = *reinterpret_cast<Qubit **>(
      __quantum__rt__array_get_element_ptr_1d(qubits, 1));

  // Place both qubits in the 1-state with X-gates.
  __quantum__qis__x(q1);
  __quantum__qis__x(q2);
  assert(*__quantum__qis__mz(q1) == 1);
  assert(*__quantum__qis__mz(q2) == 1);

  // Reset just one of the qubits and confirm the other
  // remains untouched.
  __quantum__qis__reset(q1);
  assert(*__quantum__qis__mz(q1) == 0);
  assert(*__quantum__qis__mz(q2) == 1);

  __quantum__rt__qubit_release_array(qubits);
}

Qubit *extract_qubit(Array *a, int idx) {
  auto q_raw_ptr = __quantum__rt__array_get_element_ptr_1d(a, idx);
  return *reinterpret_cast<Qubit **>(q_raw_ptr);
}

void iqft(Array *q) {
  auto nbQubits = __quantum__rt__array_get_size_1d(q);

  // Swap qubits
  for (int qIdx = 0; qIdx < nbQubits / 2; ++qIdx) {
    auto first = extract_qubit(q, qIdx);
    auto second = extract_qubit(q, nbQubits - qIdx - 1);
    __quantum__qis__swap(first, second);
  }

  for (int qIdx = 0; qIdx < nbQubits - 1; ++qIdx) {
    auto tmp = extract_qubit(q, qIdx);
    __quantum__qis__h(tmp);
    int j = qIdx + 1;
    for (int y = qIdx; y >= 0; --y) {
      const double theta = -M_PI / std::pow(2.0, j - y);
      auto first = extract_qubit(q, j);
      auto second = extract_qubit(q, y);
      __quantum__qis__cphase(theta, first, second);
    }
  }

  auto last = extract_qubit(q, nbQubits - 1);
  __quantum__qis__h(last);
}

CUDAQ_TEST(NVQIRTester, checkQPE) {
  __quantum__rt__initialize(0, nullptr);
  auto qreg = __quantum__rt__qubit_allocate_array(4);

  auto input_size = __quantum__rt__array_get_size_1d(qreg);

  // Extract the counting qubits and the state qubit
  Array *counting_qubits =
      __quantum__rt__array_slice_1d(qreg, 0, 1, input_size - 2);
  auto n_counting = __quantum__rt__array_get_size_1d(counting_qubits);
  auto state_qubit = extract_qubit(qreg, input_size - 1);

  // Put it in |1> eigenstate
  __quantum__qis__x(state_qubit);

  for (int i = 0; i < n_counting; i++) {
    auto tmp_qubit = extract_qubit(counting_qubits, i);
    __quantum__qis__h(tmp_qubit);
  }

  // run ctr-oracle operations
  for (int i = 0; i < n_counting; i++) {
    const int nbCalls = 1 << i;
    for (int j = 0; j < nbCalls; j++) {
      auto ctrl_qubit_arr =
          __quantum__rt__array_slice_1d(counting_qubits, i, 1, i);
      __quantum__qis__t__ctl(ctrl_qubit_arr, state_qubit);
    }
  }

  // Run Inverse QFT on counting qubits
  iqft(counting_qubits);

  // Measure the counting qubits
  std::vector<int> expected{1, 0, 0}, actual;
  for (int i = 0; i < n_counting; i++) {
    auto r = *__quantum__qis__mz(extract_qubit(counting_qubits, i));
    actual.push_back(r);
  }
  EXPECT_EQ(expected, actual);

  __quantum__rt__qubit_release_array(qreg);

  __quantum__rt__finalize();
}

CUDAQ_TEST(NVQIRTester, checkNisqMechanics) {
  // Library code...
  __quantum__rt__initialize(0, nullptr);

  const int shots = 100;
  cudaq::ExecutionContext ctx("sample", shots);
  __quantum__rt__setExecutionContext(&ctx);

  // Quantum Kernel Code at the QIR level
  auto qubits = __quantum__rt__qubit_allocate_array(2);
  Qubit *q1 = extract_qubit(qubits, 0);
  Qubit *q2 = extract_qubit(qubits, 1);
  __quantum__qis__h(q1);
  __quantum__qis__cnot(q1, q2);
  __quantum__qis__mz(q1);
  __quantum__qis__mz(q2);
  __quantum__rt__qubit_release_array(qubits);

  // Back to library code
  __quantum__rt__resetExecutionContext();

  cudaq::sample_result counts = ctx.result;
  int counter = 0;
  for (auto &[bits, count] :
       counts) { // std::size_t i = 0; i < counts_data.size(); i += 3) {
    EXPECT_TRUE(bits == "00" || bits == "11");
    counter += count; // counts_data[i + 2];
  }

  EXPECT_EQ(shots, counter);

  __quantum__rt__finalize();
}

CUDAQ_TEST(NVQIRTester, checkGates) {

  double oneOverSqrt2 = 1. / std::sqrt(2.0);
  std::vector<double> expectedHadamard{oneOverSqrt2, oneOverSqrt2, oneOverSqrt2,
                                       -oneOverSqrt2};
  const auto mat =
      nvqir::getGateByName<double>(nvqir::GateName::U2, {0., M_PI});
  for (std::size_t i = 0; auto m : mat) {
    EXPECT_NEAR(expectedHadamard[i++], m.real(), 1e-12);
    EXPECT_NEAR(0.0, m.imag(), 1e-12);
    std::cout << m.real() << ", " << m.imag() << "\n";
  }
  std::vector<double> expectedX{0, 1, 1, 0};
  const auto mat2 =
      nvqir::getGateByName<double>(nvqir::GateName::U3, {M_PI, 0., M_PI});

  for (std::size_t i = 0; auto m : mat2) {
    EXPECT_NEAR(expectedX[i++], m.real(), 1e-12);
    EXPECT_NEAR(0.0, m.imag(), 1e-12);
    std::cout << m.real() << ", " << m.imag() << "\n";
  }
}

CUDAQ_TEST(NVQIRTester, checkQubitAllocationFromStateVec) {
  // Library code...
  __quantum__rt__initialize(0, nullptr);

  const int shots = 1000;
  cudaq::ExecutionContext ctx("sample", shots);
  __quantum__rt__setExecutionContext(&ctx);

  // Quantum Kernel Code at the QIR level
  std::vector<cudaq::complex> bellState{M_SQRT1_2, 0.0, 0.0, M_SQRT1_2};
  Array *qubits = [](auto &state) {
    if constexpr (std::is_same_v<cudaq::complex, std::complex<double>>)
      return __quantum__rt__qubit_allocate_array_with_state_complex64(
          2, state.data());
    else
      return __quantum__rt__qubit_allocate_array_with_state_complex32(
          2, state.data());
  }(bellState);
  Qubit *q1 = extract_qubit(qubits, 0);
  Qubit *q2 = extract_qubit(qubits, 1);
  __quantum__qis__mz(q1);
  __quantum__qis__mz(q2);
  __quantum__rt__qubit_release_array(qubits);

  // Back to library code
  __quantum__rt__resetExecutionContext();

  cudaq::sample_result counts = ctx.result;
  counts.dump();
  int counter = 0;
  for (auto &[bits, count] : counts) {
    EXPECT_TRUE(bits == "00" || bits == "11");
    counter += count;
  }

  EXPECT_EQ(shots, counter);

  __quantum__rt__finalize();
}

CUDAQ_TEST(NVQIRTester, checkQubitAllocationFromRetrievedStateSimple) {
  // Library code...
  __quantum__rt__initialize(0, nullptr);
  cudaq::ExecutionContext ctx("extract-state");
  {
    __quantum__rt__setExecutionContext(&ctx);

    // Quantum Kernel Code at the QIR level
    auto qubits = __quantum__rt__qubit_allocate_array(2);
    Qubit *q1 = *reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(qubits, 0));
    Qubit *q2 = *reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(qubits, 1));

    __quantum__qis__h(q1);
    __quantum__qis__cnot(q1, q2);
    __quantum__rt__qubit_release_array(qubits);
    // Back to library code
    __quantum__rt__resetExecutionContext();
  }
  // Get the state from the context since we're about change the context.
  // Note: this is similar to passing the ctx.simulationState to the `state`
  // result in a `get_state`.
  std::unique_ptr<cudaq::SimulationState> state =
      std::move(ctx.simulationState);
  // Let's do some sampling
  const int shots = 1000;
  cudaq::ExecutionContext sampleCtx("sample", shots);
  __quantum__rt__setExecutionContext(&sampleCtx);
  auto *qubits =
      __quantum__rt__qubit_allocate_array_with_state_ptr(state.get());
  Qubit *q1 = extract_qubit(qubits, 0);
  Qubit *q2 = extract_qubit(qubits, 1);
  __quantum__qis__mz(q1);
  __quantum__qis__mz(q2);
  __quantum__rt__qubit_release_array(qubits);
  __quantum__rt__resetExecutionContext();

  cudaq::sample_result counts = sampleCtx.result;
  counts.dump();
  int counter = 0;
  for (auto &[bits, count] : counts) {
    EXPECT_TRUE(bits == "00" || bits == "11");
    counter += count;
  }

  EXPECT_EQ(shots, counter);

  __quantum__rt__finalize();
}

CUDAQ_TEST(NVQIRTester, checkQubitAllocationFromRetrievedStateExpand) {
  // Library code...
  __quantum__rt__initialize(0, nullptr);
  cudaq::ExecutionContext ctx("extract-state");
  {
    __quantum__rt__setExecutionContext(&ctx);

    // Quantum Kernel Code at the QIR level
    auto qubits = __quantum__rt__qubit_allocate_array(2);
    Qubit *q1 = *reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(qubits, 0));
    Qubit *q2 = *reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(qubits, 1));

    __quantum__qis__h(q1);
    __quantum__qis__cnot(q1, q2);
    __quantum__rt__qubit_release_array(qubits);
    // Back to library code
    __quantum__rt__resetExecutionContext();
  }
  // Get the state from the context since we're about change the context.
  // Note: this is similar to passing the ctx.simulationState to the `state`
  // result in a `get_state`.
  std::unique_ptr<cudaq::SimulationState> state =
      std::move(ctx.simulationState);
  // Let's do some sampling
  const int shots = 1000;
  cudaq::ExecutionContext sampleCtx("sample", shots);
  __quantum__rt__setExecutionContext(&sampleCtx);
  // Allocate some qubits in 0 state
  auto *someQubits = __quantum__rt__qubit_allocate_array(2);
  // Allocate some more in a specific state
  auto *qubits =
      __quantum__rt__qubit_allocate_array_with_state_ptr(state.get());
  Qubit *q1 = extract_qubit(someQubits, 0);
  Qubit *q2 = extract_qubit(someQubits, 1);
  Qubit *q3 = extract_qubit(qubits, 0);
  Qubit *q4 = extract_qubit(qubits, 1);
  // Spread the entanglement...
  __quantum__qis__cnot(q3, q1);
  __quantum__qis__cnot(q4, q2);
  __quantum__qis__mz(q1);
  __quantum__qis__mz(q2);
  __quantum__qis__mz(q3);
  __quantum__qis__mz(q4);
  __quantum__rt__qubit_release_array(qubits);
  __quantum__rt__qubit_release_array(someQubits);
  __quantum__rt__resetExecutionContext();

  cudaq::sample_result counts = sampleCtx.result;
  counts.dump();
  int counter = 0;
  // We should have a bigger GHZ state: |0000> + |1111>
  for (auto &[bits, count] : counts) {
    EXPECT_TRUE(bits == "0000" || bits == "1111");
    counter += count;
  }

  EXPECT_EQ(shots, counter);

  __quantum__rt__finalize();
}

#endif

#ifdef CUDAQ_BACKEND_DM

namespace test::hello {
struct hello_world : public ::cudaq::kraus_channel {
  hello_world(const std::vector<double> &params) {
    cudaq::real p = params[0];
    std::vector<cudaq::complex> k0v{std::sqrt(1 - p), 0, 0, std::sqrt(1 - p)},
        k1v{0, std::sqrt(p), std::sqrt(p), 0};
    push_back(cudaq::kraus_op(k0v));
    push_back(cudaq::kraus_op(k1v));
  }
  REGISTER_KRAUS_CHANNEL("test::hello::world")
};

struct adios : public ::cudaq::kraus_channel {
  constexpr static std::size_t num_parameters = 2;
  constexpr static std::size_t num_targets = 2;
  adios(const std::vector<double> &params) {
    cudaq::real p = params[0];
    cudaq::real q = params[0];
    std::vector<cudaq::complex> k0v{std::sqrt(1 - p),
                                    0,
                                    0,
                                    std::sqrt(1 - q),
                                    std::sqrt(1 - p),
                                    0,
                                    0,
                                    std::sqrt(1 - q),
                                    0,
                                    std::sqrt(p),
                                    std::sqrt(q),
                                    0,
                                    0,
                                    std::sqrt(p),
                                    std::sqrt(q),
                                    0};
    std::vector<cudaq::complex> k1v{0,
                                    std::sqrt(p),
                                    std::sqrt(q),
                                    0,
                                    0,
                                    std::sqrt(p),
                                    std::sqrt(q),
                                    0,
                                    std::sqrt(1 - p),
                                    0,
                                    0,
                                    std::sqrt(1 - q),
                                    std::sqrt(1 - p),
                                    0,
                                    0,
                                    std::sqrt(1 - q)};
    push_back(cudaq::kraus_op(k0v));
    push_back(cudaq::kraus_op(k1v));
  }
  REGISTER_KRAUS_CHANNEL("test::hello::adios")
};
} // namespace test::hello

CUDAQ_TEST(NVQIRTester, checkKrausApply) {

  const int shots = 100;
  cudaq::ExecutionContext ctx("sample", shots);
  cudaq::noise_model noise;
  noise.register_channel<test::hello::hello_world>();
  ctx.noiseModel = &noise;

  std::vector<double> params{0.2};
  __quantum__rt__setExecutionContext(&ctx);

  __quantum__rt__initialize(0, nullptr);
  auto qubits = __quantum__rt__qubit_allocate_array(1);
  Qubit *q = *reinterpret_cast<Qubit **>(
      __quantum__rt__array_get_element_ptr_1d(qubits, 0));

  __quantum__qis__x(q);
  __quantum__qis__apply_kraus_channel_double(
      test::hello::hello_world::get_key(), params.data(), params.size(),
      qubits);

  __quantum__rt__qubit_release_array(qubits);
  __quantum__rt__resetExecutionContext();

  cudaq::sample_result counts = ctx.result;
  counts.dump();
  __quantum__rt__finalize();
}

CUDAQ_TEST(NVQIRTester, checkKrausApplyGeneralUno) {

  const int shots = 100;
  cudaq::ExecutionContext ctx("sample", shots);
  cudaq::noise_model noise;
  noise.register_channel<test::hello::hello_world>();
  ctx.noiseModel = &noise;

  std::vector<double> params{0.2};
  __quantum__rt__setExecutionContext(&ctx);

  __quantum__rt__initialize(0, nullptr);
  auto qubits = __quantum__rt__qubit_allocate_array(1);
  Qubit *q = *reinterpret_cast<Qubit **>(
      __quantum__rt__array_get_element_ptr_1d(qubits, 0));

  __quantum__qis__x(q);
  __quantum__qis__apply_kraus_channel_generalized(
      1, test::hello::hello_world::get_key(), 1, 0, 1, params.data(),
      params.size(), qubits);

  __quantum__rt__qubit_release_array(qubits);
  __quantum__rt__resetExecutionContext();

  cudaq::sample_result counts = ctx.result;
  counts.dump();
  __quantum__rt__finalize();
}

CUDAQ_TEST(NVQIRTester, checkKrausApplyGeneralDue) {

  const int shots = 100;
  cudaq::ExecutionContext ctx("sample", shots);
  cudaq::noise_model noise;
  noise.register_channel<test::hello::hello_world>();
  ctx.noiseModel = &noise;

  std::vector<double> params{0.2};
  __quantum__rt__setExecutionContext(&ctx);

  __quantum__rt__initialize(0, nullptr);
  auto qubits = __quantum__rt__qubit_allocate_array(1);
  Qubit *q = *reinterpret_cast<Qubit **>(
      __quantum__rt__array_get_element_ptr_1d(qubits, 0));

  __quantum__qis__x(q);
  __quantum__qis__apply_kraus_channel_generalized(
      1, test::hello::hello_world::get_key(), 0, 1, 1, params.data(), qubits);

  __quantum__rt__qubit_release_array(qubits);
  __quantum__rt__resetExecutionContext();

  cudaq::sample_result counts = ctx.result;
  counts.dump();
  __quantum__rt__finalize();
}

CUDAQ_TEST(NVQIRTester, checkKrausApplyGeneralTre) {

  const int shots = 100;
  cudaq::ExecutionContext ctx("sample", shots);
  cudaq::noise_model noise;
  noise.register_channel<test::hello::adios>();
  ctx.noiseModel = &noise;

  std::vector<double> params{0.2, 0.4};
  __quantum__rt__setExecutionContext(&ctx);

  __quantum__rt__initialize(0, nullptr);
  auto qubits = __quantum__rt__qubit_allocate_array(2);
  Qubit *q = *reinterpret_cast<Qubit **>(
      __quantum__rt__array_get_element_ptr_1d(qubits, 0));
  Qubit *r = *reinterpret_cast<Qubit **>(
      __quantum__rt__array_get_element_ptr_1d(qubits, 0));

  __quantum__qis__x(q);
  __quantum__qis__x(r);
  __quantum__qis__apply_kraus_channel_generalized(
      1, test::hello::adios::get_key(), 1, 0, 2, params.data(), params.size(),
      qubits);

  __quantum__rt__qubit_release_array(qubits);
  __quantum__rt__resetExecutionContext();

  cudaq::sample_result counts = ctx.result;
  counts.dump();
  __quantum__rt__finalize();
}

#endif

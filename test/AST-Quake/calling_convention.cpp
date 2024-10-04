/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: x86-registered-target
// RUN: cudaq-quake %cpp_std %s | cudaq-opt | FileCheck %s

#include <cudaq.h>
#include <utility>

// Tests the host-side signatures of various spec supported kernel arguments and
// results. This file tests the x86_64 calling convention. Other architectures
// differ in their calling conventions.

struct T0 {
  void operator()() __qpu__ {}
};

struct T1 {
  void operator()(double arg) __qpu__ {}
};

struct T2 {
  void operator()(float arg) __qpu__ {}
};

struct T3 {
  void operator()(long long arg) __qpu__ {}
};

struct T4 {
  void operator()(long arg) __qpu__ {}
};

struct T5 {
  void operator()(int arg) __qpu__ {}
};

struct T6 {
  void operator()(short arg) __qpu__ {}
};

struct T7 {
  void operator()(char arg) __qpu__ {}
};

struct T8 {
  void operator()(bool arg) __qpu__ {}
};

// CHECK-LABEL:  func.func @_ZN2T0clEv(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>) {
// CHECK-LABEL:  func.func @_ZN2T1clEd(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: f64) {
// CHECK-LABEL:  func.func @_ZN2T2clEf(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: f32) {
// CHECK-LABEL:  func.func @_ZN2T3clEx(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: i64) {
// CHECK-LABEL:  func.func @_ZN2T4clEl(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: i64) {
// CHECK-LABEL:  func.func @_ZN2T5clEi(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: i32) {
// CHECK-LABEL:  func.func @_ZN2T6clEs(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: i16) {
// CHECK-LABEL:  func.func @_ZN2T7clEc(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: i8) {
// CHECK-LABEL:  func.func @_ZN2T8clEb(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: i1) {

struct R0 {
  void operator()() __qpu__ {}
};

struct R1 {
  double operator()() __qpu__ { return {}; }
};

struct R2 {
  float operator()() __qpu__ { return {}; }
};

struct R3 {
  long long operator()() __qpu__ { return {}; }
};

struct R4 {
  long operator()() __qpu__ { return {}; }
};

struct R5 {
  int operator()() __qpu__ { return {}; }
};

struct R6 {
  short operator()() __qpu__ { return {}; }
};

struct R7 {
  char operator()() __qpu__ { return {}; }
};

struct R8 {
  bool operator()() __qpu__ { return {}; }
};

// CHECK-LABEL:  func.func @_ZN2R0clEv(%arg0: !cc.ptr<i8>) {
// CHECK-LABEL:  func.func @_ZN2R1clEv(%arg0: !cc.ptr<i8>) -> f64 {
// CHECK-LABEL:  func.func @_ZN2R2clEv(%arg0: !cc.ptr<i8>) -> f32 {
// CHECK-LABEL:  func.func @_ZN2R3clEv(%arg0: !cc.ptr<i8>) -> i64 {
// CHECK-LABEL:  func.func @_ZN2R4clEv(%arg0: !cc.ptr<i8>) -> i64 {
// CHECK-LABEL:  func.func @_ZN2R5clEv(%arg0: !cc.ptr<i8>) -> i32 {
// CHECK-LABEL:  func.func @_ZN2R6clEv(%arg0: !cc.ptr<i8>) -> i16 {
// CHECK-LABEL:  func.func @_ZN2R7clEv(%arg0: !cc.ptr<i8>) -> i8 {
// CHECK-LABEL:  func.func @_ZN2R8clEv(%arg0: !cc.ptr<i8>) -> i1 {

//===----------------------------------------------------------------------===//
// structs that are less than 128 bits.

struct G0 {
  std::pair<bool, bool> operator()(std::pair<double, double>) __qpu__ {
    return {};
  }
};

struct G1 {
  std::pair<bool, char> operator()(std::pair<float, float>) __qpu__ {
    return {};
  }
};

struct G2 {
  std::pair<char, short> operator()(std::pair<long, long>,
                                    std::pair<int, double>) __qpu__ {
    return {};
  }
};

struct G3 {
  std::pair<short, short> operator()(std::pair<double, bool>) __qpu__ {
    return {};
  }
};

struct BB {
  bool _1;
  bool _2;
  bool _3;
};

struct G4 {
  std::pair<int, int> operator()(BB) __qpu__ { return {}; }
};

struct II {
  int _1;
  int _2;
  int _3;
};

struct G5 {
  std::pair<long, float> operator()(II) __qpu__ { return {}; }
};

// CHECK-LABEL:  func.func @_ZN2G0clESt4pairIddE(
// CHECK-SAME:     %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: f64,
// CHECK-SAME:     %[[VAL_2:.*]]: f64) -> i16
// CHECK-LABEL:  func.func @_ZN2G1clESt4pairIffE(
// CHECK-SAME:     %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: vector<2xf32>)
// CHECK-SAME:     -> i16
// CHECK-LABEL:  func.func @_ZN2G2clESt4pairIllES0_IidE(
// CHECK-SAME:     %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: i64,
// CHECK-SAME:     %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: f64)
// CHECK-SAME:     -> i32
// CHECK-LABEL:  func.func @_ZN2G3clESt4pairIdbE(
// CHECK-SAME:     %[[VAL_1:.*]]: !cc.ptr<i8>, %[[VAL_2:.*]]: f64,
// CHECK-SAME:     %[[VAL_3:.*]]: i64) -> i32
// CHECK-LABEL:  func.func @_ZN2G4clE2BB(
// CHECK-SAME:     %[[VAL_1:.*]]: !cc.ptr<i8>, %[[VAL_2:.*]]: i32) -> i64
// CHECK-LABEL:  func.func @_ZN2G5clE2II(
// CHECK-SAME:     %[[VAL_1:.*]]: !cc.ptr<i8>, %[[VAL_2:.*]]: i64,
// CHECK-SAME:     %[[VAL_3:.*]]: i64) -> !cc.struct<{i64, f64}>

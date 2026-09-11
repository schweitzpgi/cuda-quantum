# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_dynamic_length_list_uses_malloc():
    """A list comprehension whose length depends on a kernel parameter is
    not a compile-time constant to the AST bridge, so it must be lowered to
    a `malloc`-backed buffer, freed and reallocated via the
    `__cudaq__check_and_reallocate`/`__cudaq__check_and_free` intrinsics."""

    @cudaq.kernel
    def dynamic_len(n: int) -> int:
        l = [j for j in range(n)]
        return l[0]

    print(dynamic_len)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__dynamic_len..
# CHECK-SAME:      (%{{.*}}: i64) -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           call @malloc
# CHECK:           call @free


def test_argument_synthesis_eliminates_malloc():
    """The exact same kernel as above, but with its `n` parameter fixed by
    argument synthesis: `stack-allocate-const-lists` (run immediately after
    `argument-synthesis` in the pipeline) must recognize that the list's
    length is now a compile-time constant and convert its buffer back into
    a plain, stack-allocated, `malloc`/`free`-free local."""

    @cudaq.kernel
    def dynamic_len_synth(n: int) -> int:
        l = [j for j in range(n)]
        return l[0]

    print(cudaq.synthesize(dynamic_len_synth, 5))


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__dynamic_len_synth..
# CHECK-SAME:      (%{{.*}}: i64) -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[ALLOCA:.*]] = cc.alloca !cc.array<i64 x 5>
# CHECK-NOT:       call @malloc
# CHECK-NOT:       call @free
# CHECK:           return

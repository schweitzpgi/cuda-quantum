# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(anyon)
if (AWSSDK_ROOT)
  add_subdirectory(braket)
endif()
add_subdirectory(infleqtion)
add_subdirectory(ionq)
add_subdirectory(iqm)
add_subdirectory(oqc)
add_subdirectory(quantinuum)
if (CUDAQ_ENABLE_QUANTUM_MACHINES_BACKEND)
  add_subdirectory(quantum_machines)
endif()

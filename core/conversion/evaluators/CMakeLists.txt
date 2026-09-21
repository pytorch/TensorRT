# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set(sub_lib_name "evaluators")

target_sources(${lib_name}
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/NodeEvaluatorRegistry.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/aten.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/eval_util.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/prim.cpp"
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/evaluators.h"
)

# Install headers
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/conversion/${sub_lib_name}")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture metadata keys shared by tracing and compilation."""

TOKEN_KEY = "torch_tensorrt_region_token"
NAME_KEY = "torch_tensorrt_region_name"
BACKEND_KEY = "torch_tensorrt_region_backend"
ROUTE_KEY = "torch_tensorrt_region_route"
# The kernel's registry key and the custom operator's name; "" when unused.
PLUGIN_KEY = "torch_tensorrt_region_plugin"
OPERATOR_KEY = "torch_tensorrt_region_operator"
# Set on regions that a ``regions=`` rule selected, not the model's source.
SELECTED_KEY = "torch_tensorrt_region_selected"
SUPPORTED_BACKENDS = ("eager", "inductor")

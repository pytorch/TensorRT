# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture metadata keys shared without registering marker operators."""

TOKEN_KEY = "torch_tensorrt_region_token"
NAME_KEY = "torch_tensorrt_region_name"
BACKEND_KEY = "torch_tensorrt_region_backend"
ROUTE_KEY = "torch_tensorrt_region_route"
PLUGIN_KEY = "torch_tensorrt_region_plugin"
SUPPORTED_BACKENDS = ("eager", "inductor")

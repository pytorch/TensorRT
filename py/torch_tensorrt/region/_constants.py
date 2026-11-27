# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Constants shared by tracing and compilation."""

# Set on the end marker of a region that a ``regions=`` rule selected.
SELECTED_KEY = "torch_tensorrt_region_selected"
SUPPORTED_BACKENDS = ("eager", "inductor")

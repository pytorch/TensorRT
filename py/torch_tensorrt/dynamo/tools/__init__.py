# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch_tensorrt.dynamo.tools.weight_mapping import (  # noqa: F401
    MappedWeights,
    map_weights,
    release_mapped_storage,
    save_weights_for_mapping,
)

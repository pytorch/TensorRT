# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch_tensorrt
from torch_tensorrt.dynamo.runtime._ResourceAllocator import (  # noqa: F401
    ResourceAllocationStrategy,
)
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (  # noqa: F401
    TorchTensorRTModule,
)

if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
    from torch_tensorrt.dynamo.runtime.meta_ops.register_meta_ops import *

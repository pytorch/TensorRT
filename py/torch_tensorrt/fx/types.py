# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Sequence, Tuple

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt

if hasattr(trt, "__version__"):
    TRTNetwork = trt.INetworkDefinition
    TRTTensor = trt.tensorrt.ITensor
    TRTLayer = trt.ILayer
    TRTPluginFieldCollection = trt.PluginFieldCollection
    TRTPlugin = trt.IPluginV3
    TRTDataType = trt.DataType
    TRTElementWiseOp = trt.ElementWiseOperation
else:
    TRTNetwork = "trt.INetworkDefinition"
    TRTTensor = "trt.tensorrt.ITensor"
    TRTLayer = "trt.ILayer"
    TRTPluginFieldCollection = "trt.PluginFieldCollection"
    TRTPlugin = "trt.IPluginV3"
    TRTDataType = "trt.DataType"
    TRTElementWiseOp = "trt.ElementWiseOperation"

Shape = Sequence[int]
ShapeRange = Tuple[Shape, Shape, Shape]

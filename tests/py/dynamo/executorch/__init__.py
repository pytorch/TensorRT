# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from pkgutil import extend_path

# CI runs these tests from tests/py/dynamo, where this package would otherwise
# shadow the installed ExecuTorch package and hide executorch.exir.
__path__ = extend_path(__path__, __name__)

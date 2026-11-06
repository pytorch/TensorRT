# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os

from torch._inductor.runtime.cache_dir_utils import cache_dir

# AOTInductor builds a graph in a folder named after the graph's code and deletes a fixed-name
# object file there, locking only its own build. Two pytest-xdist workers building graphs with
# the same code can delete each other's file, so give each worker its own Inductor cache.
_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_dir(), _worker)

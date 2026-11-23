# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# This module imports nothing from the package, so the lowering passes, the converters and the
# partitioners can all import it at the top.

# The number of key values of a stable sort, in its node.meta. The annotate_sort_num_keys pass
# sets it at the end of post_lowering, only when it proves that every key lies in [0, N), and
# the partitioners remove it from a sort they do not put in the engine that makes its keys. The
# sort converter trusts it for correctness. The engine cache does not hash it: it is the size of
# a dim of the input of a topk, argmax or argmin in the same engine, so the hashed graph, input
# shapes and weight shapes already fix it.
NUM_SORT_KEYS = "_num_sort_keys"

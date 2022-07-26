import os
import sys
from typing import Any

import torch.fx
from torch.fx.passes import shape_prop


class SuppressStderrPrints:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr


class AccShapeProp(shape_prop.ShapeProp):
    """
    Similar to standard shape prop, but if any node that is run with standard shape prop
    fails then it tries to upconvert any fp16 inputs to fp32, rerun shape prop, and then
    downconvert fp32 results back to fp16.

    Note that we currently mostly only look for/support up/down conversion for nodes
    with tensor outputs, but this is likely fine for most cases. Additionally the base
    shape_prop works for many ops with fp16, such as tensor.cat, tensor slice, tensor.to
    dtype conversion, etc.

    """

    def run_node(self, n: torch.fx.Node) -> Any:
        # First try running shape_prop with the original inputs.
        with SuppressStderrPrints():
            try:
                return super().run_node(n)
            except Exception:
                pass

        # Base shape_prop failed, so temporarily upconvert the node's fp16 inputs in env
        # and retry. For now just support upconverting Tensor outputs.
        orig_dtype_env = []
        for in_node in n.all_input_nodes:
            in_ten = self.env[in_node]
            if isinstance(in_ten, torch.Tensor) and in_ten.dtype == torch.float16:
                orig_dtype_env.append((in_node, in_ten))
                self.env[in_node] = in_ten.clone().to(dtype=torch.float)

        # Now try running again with upconverted fp32 input tensor in env.
        result = super().run_node(n)

        # Now that we succeeded, assume it's thanks to upconverting. Therefore we
        # downconvert fp32 tensor results to fp16.
        if isinstance(result, torch.Tensor) and result.dtype == torch.float:
            result = result.to(dtype=torch.float16)
            self.env[n] = result
            n.meta["tensor_meta"] = n.meta["tensor_meta"]._replace(dtype=torch.float16)

        # Finally, restore the original env back to fp16 for any upconverted tensors.
        for in_node, in_ten in orig_dtype_env:
            self.env[in_node] = in_ten

        return result

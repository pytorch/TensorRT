import os
import sys
from typing import Any

import torch.fx

import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
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

    def _run_node(self, n: torch.fx.Node) -> Any:
        # Run embedding bag ops with XL weights in a customized way, see
        # docstring for self.run_embedding_bag for more details
        if (
            n.target
            in {
                acc_ops.embedding_bag,
                acc_ops.embedding_bag_4bit_rowwise_offsets,
                acc_ops.embedding_bag_byte_rowwise_offsets,
            }
            and n.kwargs["weight"].target == acc_ops.xl_weight
        ):
            return self.run_embedding_bag(n)
        return super().run_node(n)

    def run_node(self, n: torch.fx.Node) -> Any:
        # First try running shape_prop with the original inputs.
        with SuppressStderrPrints():
            try:
                return self._run_node(n)
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
        result = self._run_node(n)

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

    def run_embedding_bag(self, n: torch.fx.Node) -> Any:
        """
        EmbeddingBag with XL Weights of shape (num_embeddings, embedding_dim)
        are replaced with smaller proxies of shape
        (acc_ops.PROXY_EMBEDDING_SIZE, embedding_dim) during tracing. This can
        cause index out of bounds issues when sample inputs lead to the
        embedding bag op indexing into the first dimension of the weight tensor
        which it expects to be bigger than it is during tracing.
        """
        if n.target == acc_ops.embedding_bag:
            indices = n.kwargs["input"]
        else:
            indices = n.kwargs["indices"]

        # Replace indices with zeros of same shape and dtype
        indices_tensor = self.env[indices]
        indices_zeros = torch.zeros_like(indices_tensor, dtype=indices_tensor.dtype)
        self.env[indices] = indices_zeros

        # Run node
        result = super().run_node(n)

        # Restore indices
        self.env[indices] = indices_tensor

        return result

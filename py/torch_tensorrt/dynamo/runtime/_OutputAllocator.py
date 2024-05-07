import logging

import tensorrt as trt
import torch

logger = logging.getLogger(__name__)


class OutputAllocator(trt.IOutputAllocator):
    def __init__(self):
        trt.IOutputAllocator.__init__(self)
        self.buffers = {}
        self.shapes = {}

    def reallocate_output(self, tensor_name, memory, size, alignment):
        shape = (size,)
        if tensor_name not in self.buffers:
            self.buffers[tensor_name] = torch.empty(
                shape, dtype=torch.uint8, device="cuda"
            )
        else:
            self.buffers[tensor_name] = self.buffers[tensor_name].resize_(shape)
        logger.debug(
            f"Reallocated output tensor: {tensor_name} to: {self.buffers[tensor_name]}"
        )
        return self.buffers[tensor_name].data_ptr()

    def notify_shape(self, tensor_name, shape):
        self.shapes[tensor_name] = tuple(shape)

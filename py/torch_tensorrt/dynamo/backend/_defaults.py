from torch_tensorrt.fx.utils import LowerPrecision


PRECISION = LowerPrecision.FP32
DEBUG = False
MAX_WORKSPACE_SIZE = 20 << 30
MIN_BLOCK_SIZE = 5

from torch_tensorrt.fx.utils import LowerPrecision


PRECISION = LowerPrecision.FP32
DEBUG = False
WORKSPACE_SIZE = 0
MIN_BLOCK_SIZE = 5
PASS_THROUGH_BUILD_FAILURES = False

from torch_tensorrt.fx.utils import LowerPrecision


PRECISION = LowerPrecision.FP32
DEBUG = False
MAX_WORKSPACE_SIZE = 20 << 30
MAX_NUM_TRT_ENGINES = 10

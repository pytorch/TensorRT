from setuptools import setup

setup(
    name="torch_tensorrt_fx2trt",
    version="0.1",
    description="Torch-FX to TensorRT Converter",
    author="PyTorch Team",
    packages=[
        "torch_tensorrt.fx",
        "torch_tensorrt.fx.converters",
        "torch_tensorrt.fx.passes",
        "torch_tensorrt.fx.tools",
        "torch_tensorrt.fx.tracer.acc_tracer",
    ],
    package_dir={
        "torch_tensorrt.fx": "../",
        "torch_tensorrt.fx.converters": "../converters",
        "torch_tensorrt.fx.passes": "../passes",
        "torch_tensorrt.fx.tools": "../tools",
        "torch_tensorrt.fx.tracer.acc_tracer": "../tracer/acc_tracer",
    },
)

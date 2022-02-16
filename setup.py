#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='fx2trt_oss',
    version='0.1',
    description='Torch-FX to TensorRT Converter',
    author='PyTorch Team',
    packages=['fx2trt_oss.fx', 'fx2trt_oss.fx.converters', 'fx2trt_oss.fx.passes', 'fx2trt_oss.fx.tools', 'fx2trt_oss.tracer.acc_tracer'],
    package_dir={'fx2trt_oss.fx': 'fx',
        'fx2trt_oss.fx.converters': 'fx/converters',
        'fx2trt_oss.fx.passes': 'fx/passes',
        'fx2trt_oss.fx.tools': 'fx/tools',
        'fx2trt_oss.tracer.acc_tracer': 'tracer/acc_tracer',
    },
)

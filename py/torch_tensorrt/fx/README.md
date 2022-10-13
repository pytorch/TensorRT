FX2TRT is merged as FX module in Torch-TensorRT

- The user guide is in [link](../../../docsrc/tutorials/getting_started_with_fx_path.rst#installation)
- The examples are moved to [link](../../../examples/fx)

* Method 1. Follow the instrucions for Torch-TensorRT
* Method 2. To install FX path only (Python path) and avoid the C++ build for torchscript path
`
    $ conda create --name python_env python=3.8
    $ conda activate python_env
    # Recommend to install PyTorch 1.12 and later
    $ conda install pytorch torchvision torchtext cudatoolkit=11.3 -c pytorch-nightly
    # Install TensorRT python package
    $ pip3 install nvidia-pyindex
    $ pip3 install nvidia-tensorrt==8.2.4.2
    $ git clone https://github.com/pytorch/TensorRT.git
    $ cd TensorRT/py && python setup.py install --fx-only && cd ..
    $ pyton -c "import torch_tensorrt.fx"
    # Test an example by
    $ python py/torch_tensorrt/fx/example/lower_example.py
`

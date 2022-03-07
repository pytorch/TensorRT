# FX2TRT
This package provide pure eager-mode tooling to convert a PyTorch nn.Module to a TensorRT engine.

## Installation
First, let's install PyTorch.
```
conda install -y pytorch cudatoolkit=11.3 -c pytorch-nightly
```
Then, you need to install your TensorRT and it's pythin binding
```
tar -xzvf TensorRT-8.2.1.8.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz
export LD_LIBRARY_PATH=$HOME/TensorRT-8.2.1.8/lib:$HOME/TensorRT-8.2.1.8/targets/x86_64-linux-gnu/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
cd TensorRT-8.2.1.8/python
python3 -m pip install tensorrt-8.2.1.8-cp36-none-linux_x86_64.whl
```
Then, it's simply as this.
```
cd fx2trt
python setup.py install
```
## Test
Follow instruction in [pytorch/benchmark](https://github.com/pytorch/benchmark) to setup some benchmarks.

Then try using `--fx2trt` for individual cases, e.g.
```
cd benchmark
python run.py resnet50 -d cuda -t eval -m eager --fx2trt
```
And you should expect TensorRT logs being printed and the case ran through without error.

# Jupyter demo notebooks
This folder contains demo notebooks for Torch-TensorRT.

## 1. Requirements

The most convenient way to run these notebooks is via a docker container, which provides a self-contained, isolated and re-producible environment for all experiments.

First, clone the repository:

```
git clone https://github.com/NVIDIA/Torch-TensorRT
```

Next, build the NVIDIA Torch-TensorRT container (from repo root):

```
docker build -t torch_tensorrt -f ./docker/Dockerfile.21.06 .
```

Then launch the container with:

```
docker run --runtime=nvidia -it --rm --ipc=host --net=host torch_tensorrt
```

Within the docker interactive bash session, start Jupyter with

```
jupyter notebook --allow-root --ip 0.0.0.0 --port 8888
```

And navigate a web browser to the IP address or hostname of the host machine
at port 8888: ```http://[host machine]:8888```

Use the token listed in the output from running the jupyter command to log
in, for example:

```http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b```


Within the container, the notebooks themselves are located at `/workspace/Torch-TensorRT/notebooks`.

## 2. Notebook list

- [lenet-getting-started.ipynb](lenet-getting-started.ipynb): simple example on a LeNet network.
- [ssd-object-detection-demo.ipynb](ssd-object-detection-demo.ipynb): demo for compiling a pretrained SSD model using Torch-TensorRT.
- [Resnet50-example.ipynb](Resnet50-example.ipynb): demo on the ResNet-50 network.
- [vgg-qat.ipynb](vgg-qat.ipynb): Quantization Aware Trained models in INT8 using Torch-TensorRT

```python

```

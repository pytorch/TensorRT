# Jupyter demo notebooks
This folder contains demo notebooks for Torch-TensorRT.

## 1. Requirements

The most convenient way to run these notebooks is via a docker container, which provides a self-contained, isolated and re-producible environment for all experiments.

First, clone the repository:

```
git clone https://github.com/pytorch/TensorRT
```

Next, navigate to the repo's root directory:

```
cd Torch-TensorRT
```

### a. Using the NGC PyTorch container

At this point, we recommend pulling the [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
from [NVIDIA GPU Cloud](https://catalog.ngc.nvidia.com/) as follows:

```
docker pull nvcr.io/nvidia/pytorch:22.05-py3
```

Replace ```22.05``` with a different string in the form ```yy.mm```,
where ```yy``` indicates the last two numbers of a calendar year, and
```mm``` indicates the month in two-digit numerical form, if you wish
to pull a different version of the container.

The NGC PyTorch container ships with the Torch-TensorRT tutorial notebooks.
Therefore, you can run the container and the notebooks therein without
mounting the repo to the container. To do so, run

```
docker run --gpus=all --rm -it --net=host --ipc=host \
--ulimit memlock=-1 --ulimit stack=67108864 \
nvcr.io/nvidia/pytorch:22.05-py3 bash
```

If, however, you wish for your work in the notebooks to persist, use the
```-v``` flag to mount the repo to the container as follows:

```
docker run --gpus=all --rm -it -v $PWD:/Torch-TensorRT \
--net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
nvcr.io/nvidia/pytorch:22.05-py3 bash
```

### b. Building a Torch-TensorRT container from source

Alternatively, to build the container from source, run

```
docker build -t torch_tensorrt -f ./docker/Dockerfile .
```

To run this container, enter the following command:

```
docker run --gpus=all --rm -it -v $PWD:/Torch-TensorRT \
--net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
torch_tensorrt:latest bash
```

### c. Running the notebooks inside the container

Within the docker interactive bash session, proceed to the notebooks.
To use the notebooks which ship with the container, run

```
cd /workspace/examples/torch_tensorrt/notebooks
```

If, however, you mounted the repo to the container, run

```
cd /Torch-TensorRT/notebooks
```

Once you have entered the appropriate ```notebooks``` directory, start Jupyter with

```
jupyter notebook --allow-root --ip 0.0.0.0 --port 8888
```

And navigate a web browser to the IP address or hostname of the host machine
at port 8888: ```http://[host machine]:8888```

Use the token listed in the output from running the jupyter command to log
in, for example:

```http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b```


Within the container, the notebooks themselves are located at `/Torch-TensorRT/notebooks`.

## 2. Notebook list

- [lenet-getting-started.ipynb](lenet-getting-started.ipynb): simple example on a LeNet network.
- [ssd-object-detection-demo.ipynb](ssd-object-detection-demo.ipynb): demo for compiling a pretrained SSD model using Torch-TensorRT.
- [Resnet50-example.ipynb](Resnet50-example.ipynb): demo on the ResNet-50 network.
- [vgg-qat.ipynb](vgg-qat.ipynb): Quantization Aware Trained models in INT8 using Torch-TensorRT
- [EfficientNet-example.ipynb](EfficientNet-example.ipynb): Simple use of 3rd party PyTorch model library
- [CitriNet-example.ipynb](CitriNet-example.ipynb): Optimizing the Nemo Citrinet acoustic model

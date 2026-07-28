# Building a Torch-TensorRT container

* Use `Dockerfile` to build a container which provides the exact development environment that our main branch is usually tested against.

* The `Dockerfile` currently uses <a href="https://github.com/bazelbuild/bazelisk">Bazelisk</a> to select the Bazel version, and uses the exact library versions of Torch and CUDA listed in <a href="https://github.com/pytorch/TensorRT#dependencies">dependencies</a>.
  * The desired versions of TensorRT must be specified as build-args, with major and minor versions as in: `--build-arg TENSORRT_VERSION=a.b`
  * [**Optional**] The desired base image be changed by explicitly setting a base image, as in `--build-arg BASE_IMG=nvidia/cuda:11.8.0-devel-ubuntu22.04`, though this is optional.
  * [**Optional**] Additionally, the desired Python version can be changed by explicitly setting a version, as in `--build-arg PYTHON_VERSION=3.11`, though this is optional as well.

* This `Dockerfile` installs `cxx11-abi` versions of Pytorch and builds Torch-TRT using `cxx11-abi` libtorch as well. As of torch 2.7, torch requires `cxx11-abi` for all CUDA 11.8, 12.4, 12.6, and later versions.

### Dependencies

* Install nvidia-docker by following https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

### Instructions

- The example below uses TensorRT 10.13.2.6
- See <a href="https://github.com/pytorch/TensorRT#dependencies">dependencies</a> for a list of current default dependencies.

> From root of Torch-TensorRT repo

Build:
```
DOCKER_BUILDKIT=1 docker build --build-arg TENSORRT_VERSION=10.13.2 -f docker/Dockerfile -t torch_tensorrt:latest .
```

Run:
```
nvidia-docker run --gpus all -it --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --name=torch_tensorrt --ipc=host --net=host torch_tensorrt:latest
```

Test:


You can run any converter test to verify if Torch-TRT built successfully inside the container. Once you launch the container, you can run
```
bazel test //tests/core/conversion/converters:test_activation --compilation_mode=opt --test_output=summary --config use_precompiled_torchtrt
```

* `--config use_precompiled_torchtrt` : Indicates bazel to use pre-installed Torch-TRT library to test an application.

### Pytorch NGC containers

We also ship Torch-TensorRT in <a href="https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch">Pytorch NGC containers </a>. Release notes for these containers can be found <a href="https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html">here</a>. Check out `release/ngc/23.XX` branch of Torch-TensorRT for source code that gets shipped with `23.XX` version of Pytorch NGC container.

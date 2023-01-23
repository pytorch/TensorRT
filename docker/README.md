# Building a Torch-TensorRT container

* Use `Dockerfile` to build a container which provides the exact development environment that our master branch is usually tested against.

* `Dockerfile` currently uses the exact library versions (Torch, CUDA, CUDNN, TensorRT) listed in <a href="https://github.com/pytorch/TensorRT#dependencies">dependencies</a> to build Torch-TensorRT.

* This `Dockerfile` installs `pre-cxx11-abi` versions of Pytorch and builds Torch-TRT using `pre-cxx11-abi` libtorch as well.
Note: To install `cxx11_abi` version of Torch-TensorRT, enable `USE_CXX11=1` flag so that `dist-build.sh` can build it accordingly.

### Dependencies

* Install nvidia-docker by following https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

### Instructions

> From root of Torch-TensorRT repo

Build:
```
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t torch_tensorrt:latest .
```

Run:
```
nvidia-docker run --gpus all -it --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --name=torch_tensorrt --ipc=host --net=host torch_tensorrt:latest
```

Test:


You can run any converter test to verify if Torch-TRT built sucessfully inside the container. Once you launch the container, you can run
```
bazel test //tests/core/conversion/converters:test_activation --compilation_mode=opt --test_output=summary --config use_precompiled_torchtrt --config pre_cxx11_abi
```

* `--config use_precompiled_torchtrt` : Indicates bazel to use pre-installed Torch-TRT library to test an application.
* `--config pre_cxx11_abi` : This flag ensures `bazel test` uses `pre_cxx11_abi` version of `libtorch`. Use this flag corresponding to the ABI format of your Torch-TensorRT installation.

### Pytorch NGC containers

We also ship Torch-TensorRT in <a href="https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch">Pytorch NGC containers </a>. Release notes for these containers can be found <a href="https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html">here</a>. Check out `release/ngc/23.XX` branch of Torch-TensorRT for source code that gets shipped with `23.XX` version of Pytorch NGC container.
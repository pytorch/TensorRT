# Building a Torch-TensorRT container

Use `Dockerfile` to build a container which provides the exact development environment that our master branch is usually tested against.
`Dockerfile` currently uses the following library versions to build Torch-TensorRT.

| Library  | Version |
| ------------- | ------------- |
| CUDA  | 11.7.1  |
| CUDNN  | 8.4.1  |
| TensorRT  | 8.4.3.1  |
| Pytorch  | 1.13.0.dev20221006+cu117  |
| torchvision  | 0.15.0.dev20221006+cu117  |

This `Dockerfile` installs `pre-cxx11-abi` versions of Pytorch and builds Torch-TRT using `pre-cxx11-abi` libtorch as well.

### Dependencies

Install nvidia-docker by following https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

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

### Notes

We also ship Torch-TensorRT in <a href="https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch">Pytorch NGC containers </a>. Release notes for these containers can be found <a href="https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html">here</a>. Check out `release/ngc/22.XX` branch of Torch-TensorRT for source code that gets shipped with `22.XX` version of Pytorch NGC container.

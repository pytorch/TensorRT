# Building a Torch-TensorRT container

### Install Docker and NVIDIA Container Toolkit

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Build Container

> From root of Torch-TensorRT repo

```
# Build:
DOCKER_BUILDKIT=1 docker build --build-arg BASE={TensorRT Base Container Version} -f docker/Dockerfile -t torch_tensorrt1.0:latest .

# Run:
docker run --gpus all -it \
	--shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--name=torch_tensorrt1.0 --ipc=host --net=host torch_tensorrt1.0:latest
```

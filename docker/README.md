
## Use the container for pytorch1.10+cuda11.1+trt8.0.3.4

```
# Build:
docker build -f docker/Dockerfile -t torch_tensorrt1.0:latest .

# Run:
docker run --gpus all -it \
	--shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--name=torch_tensorrt1.0 --ipc=host --net=host torch_tensorrt1.0:latest
```

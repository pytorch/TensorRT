
## Use the container for pytorch1.7+cuda11.1+trt7.2.1 (with docker ≥ 19.03)

```
# Build:
docker build -f docker/Dockerfile.20.10 -t trtorch:pytorch1.7-cuda11.1-trt7.2.1 .

# Run:
docker run --gpus all -it \
	--shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--name=trtorch --ipc=host --net=host trtorch:pytorch1.7-cuda11.1-trt7.2.1
```
## Use the container for pytorch1.6+cuda11.0+trt7.1.3 (with docker ≥ 19.03)

```
# Build:
docker build -f docker/Dockerfile.20.07 -t trtorch:pytorch1.6-cuda11.0-trt7.1.3 .

# Run:
docker run --gpus all -it \
	--shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--name=trtorch --ipc=host --net=host trtorch:pytorch1.6-cuda11.0-trt7.1.3
```
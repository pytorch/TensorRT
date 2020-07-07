# Jupyter demo notebooks
This folder contains demo notebooks for the TRTorch.

## 1. Requirements

The most convenient way to run these notebooks is via a docker container, which provides a self-contained, isolated and re-producible environment for all experiments. 

First, clone the repository:

```
git clone https://github.com/NVIDIA/TRTorch
```

Next, build the NVIDIA TRTorch container:

```
docker build -t trtorch -f Dockerfile.notebook .  
```

Then launch the container with:

```
docker run --runtime=nvidia -it --rm --ipc=host --net=host trtorch 
```
where `/path/to/dataset` is the path on the host machine where the data was/is to be downloaded. More on data set preparation in the next section. `/path/to/results` is wher the trained model will be stored.

Within the docker interactive bash session, start Jupyter with

```
jupyter notebook --allow-root --ip 0.0.0.0 --port 8888
```

Then open the Jupyter GUI interface on your host machine at http://localhost:8888. Within the container, this notebook itself is located at `/workspace/TRTorch/notebooks`.

## 2. Notebook list

- [LeNet-example.ipynb](LeNet-example.ipynb): simple example on a LeNet network.

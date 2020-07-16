# Jupyter demo notebooks
This folder contains demo notebooks for TRTorch.

## 1. Requirements

The most convenient way to run these notebooks is via a docker container, which provides a self-contained, isolated and re-producible environment for all experiments.

First, clone the repository:

```
git clone https://github.com/NVIDIA/TRTorch
```

Next, build the NVIDIA TRTorch container (from repo root):

```
docker build -t trtorch -f notebooks/Dockerfile.notebook .
```

Then launch the container with:

```
docker run --runtime=nvidia -it --rm --ipc=host --net=host trtorch
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


Within the container, the notebooks themselves is located at `/workspace/TRTorch/notebooks`.

## 2. Notebook list

- [LeNet-example.ipynb](LeNet-example.ipynb): simple example on a LeNet network.

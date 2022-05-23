# Torch-TensorRT to Triton

This README showcases how to deploy a simple ResNet model accelerated by using Torch-TensorRT on Triton Inference Server.

## Step 1: Optimize your model with Torch-TensorRT

If you are unfamiliar with Torch-TensorRT please refer this [video](https://www.youtube.com/watch?v=TU5BMU6iYZ0&ab_channel=NVIDIADeveloper). The first step in this pipeline is to accelerate your model. While using PyTorch as your framework of choice for training, you can either use TensorRT or Torch-TensorRT, depending on your model's operations.

For using Torch-TensorRT, let's first pull the NGC PyTorch Docker container,, which comes installed with both TensorRT and Torch-TensorRT. You may need to create an account and get the API key from [here](https://ngc.nvidia.com/setup/). Sign up and login with your key (follow the instructions [here](https://ngc.nvidia.com/setup/api-key) after signing up).

```
# <xx.xx> is the yy:mm for the publishing tag for NVIDIA's Pytorch 
# container; eg. 22.04

docker run -it --gpus all -v /path/to/this/folder:/resnet50_eg nvcr.io/nvidia/pytorch:<xx.xx>-py3
```

We have already made a sample to use Torch-TensorRT: `torch_trt_resnet50.py`. This sample downloads a ResNet model from Torchhub and then optimizes it with Torch-TensorRT. For more examples, visit our [Github Repository](https://github.com/NVIDIA/Torch-TensorRT/tree/master/notebooks).

```
python torch_trt_resnet50.py

# you can exit out of this container now
exit
```

## Step 2: Set Up Triton Inference Server

If you are new to the Triton Inference Server and want to learn more, we highly recommend to checking our [Github Repository](https://github.com/triton-inference-server).

To use Triton, we need to make a model repository. The structure of the repository should look something like this:
```
model_repository
|
+-- resnet50
    |
    +-- config.pbxt
    +-- 1
        |
        +-- model.pt
```

A sample model configuration of the model is included with this demo as `config.pbtxt`. If you are new to Triton, we highly encourage you to check out this [section of our documentation](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md) for more details. Once you have the model repository setup, it is time to launch the Triton server! You can do that with the docker command below.
```
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models
```

## Step 3: Using a Triton Client to Query the Server

Download an example image to test inference.

```
wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```

Install dependencies.
```
pip install torchvision
pip install attrdict
pip install nvidia-pyindex
pip install tritonclient[all]
```

Run client
```
python3 triton_client.py
```
The output of the same should look like below:
```
[b'12.468750:90' b'11.523438:92' b'9.664062:14' b'8.429688:136'
 b'8.234375:11']
```
The output format here is `<confidence_score>:<classification_index>`. To learn how to map these to the label names and more, refer to our [documentation](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md).
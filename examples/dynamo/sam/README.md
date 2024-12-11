# SAM2 model compilation using Torch-TensorRT

This example illustrates the state of the art model `Segment Anything Model 2` optimized using
Torch-TensorRT.

## Dependencies
Install the following dependencies before running the model.

```py
pip install -r ../requirements.txt
```

## Model execution
Run the following script to produce output masks after compiling with Torch-TensorRT
```py
python run_sam.py
```

## Inputs and Outputs

Here is the input image and the predicted masks by Torch-TRT.

The output masks are as shown below
![alt text](https://github.com/pytorch/TensorRT/blob/sam/examples/dynamo/sam/truck.jpg?raw=true)

Mask 1                     |  Mask 2                   |  Mask 3
:-------------------------:|:-------------------------:|:-------------------------:|
![](https://github.com/pytorch/TensorRT/blob/sam/examples/dynamo/sam/Torch-TRT_output_mask_1.png?raw=true)  |  ![](https://github.com/pytorch/TensorRT/blob/sam/examples/dynamo/sam/Torch-TRT_output_mask_2.png?raw=true)   ![](https://github.com/pytorch/TensorRT/blob/sam/examples/dynamo/sam/Torch-TRT_output_mask_3.png?raw=true)



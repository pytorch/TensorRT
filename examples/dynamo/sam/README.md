# SAM2 model compilation using Torch-TensorRT

This example illustrates the state of the art model `Segment Anything Model 2` optimized using
Torch-TensorRT.

## Dependencies
Install the following dependencies before running the model.

```py
pip install -r ../requirements.txt
```

Certain custom modifications are required to ensure the model is exported successfully. To apply these changes, please install SAM2 using the following fork : https://github.com/chohk88/sam2/tree/torch-trt (<a href="https://github.com/chohk88/sam2/tree/torch-trt?tab=readme-ov-file#installation">Installation</a>)

## Model execution
Run the following script to produce output masks after compiling with Torch-TensorRT
```py
python run_sam.py
```

## Inputs and Outputs

Here is the input image and the predicted masks by Torch-TRT.

<p align="center">
  <img src="https://github.com/pytorch/TensorRT/blob/sam/examples/dynamo/sam/truck.jpg?raw=true" alt="Input image" width="400" height="400">
  <img src="https://github.com/pytorch/TensorRT/blob/sam/examples/dynamo/sam/Torch-TRT_output_mask_1.png?raw=true" alt="Predicted mask 1" width="400" height="400">
</p>
<p align="center">
  <img src="https://github.com/pytorch/TensorRT/blob/sam/examples/dynamo/sam/Torch-TRT_output_mask_2.png?raw=true" alt="Predicted mask 2" width="400" height="400">
  <img src="https://github.com/pytorch/TensorRT/blob/sam/examples/dynamo/sam/Torch-TRT_output_mask_3.png?raw=true" alt="Predicted mask 3" width="400" height="400">
</p>

## References:

1) <a href="https://arxiv.org/pdf/2408.00714">SAM 2: Segment Anything in Images and Videos </a>
2) SAM2 github repository : https://github.com/facebookresearch/sam2/tree/main



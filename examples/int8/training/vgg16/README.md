# VGG16 Trained on CIFAR10

This is a recipe to train a VGG network on CIFAR10 to use with the Torch-TensorRT PTQ example.

## Prequisites

```
pip3 install -r requirements.txt --user
```

To perform quantization aware training, please install NVIDIA's <a href="https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization">pytorch quantization toolkit</a>

## Training

The following recipe should get somewhere between 89-92% accuracy on the CIFAR10 testset

```
python3 main.py --lr 0.01 --batch-size 128 --drop-ratio 0.15 --ckpt-dir $(pwd)/vgg16_ckpts --epochs 100
```

> 545 was the seed used in testing

You can monitor training with tensorboard, logs are stored by default at `/tmp/vgg16_logs`

### Quantization Aware Fine Tuning (for trying out QAT Workflows)

To perform quantization aware training, it is recommended that you finetune your model obtained from previous step with quantization layers.

```
python3 finetune_qat.py --lr 0.01 --batch-size 128 --drop-ratio 0.15 --ckpt-dir $(pwd)/vgg16_ckpts --start-from 100 --epochs 110
```

Please expect to see some warnings during `finetune_qat.py` execution as follows. You can ignore these warnings and continue to finetune the model.

```python
E0810 20:56:22.967188 139913188632384 tensor_quantizer.py:121] Fake quantize mode doesn't use scale explicitly!
E0810 20:56:22.967244 139913188632384 tensor_quantizer.py:121] Fake quantize mode doesn't use scale explicitly!
E0810 20:56:22.967279 139913188632384 tensor_quantizer.py:135] step_size is undefined under dynamic amax mode!
E0810 20:56:22.967308 139913188632384 tensor_quantizer.py:135] step_size is undefined under dynamic amax mode!
python3.6/site-packages/pytorch_quantization-2.1.0-py3.6-linux-x86_64.egg/pytorch_quantization/tensor_quant.py:322: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
W0810 20:56:24.370095 139913188632384 tensor_quantizer.py:174] Disable MaxCalibrator
W0810 20:56:24.370152 139913188632384 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
W0810 20:56:24.370185 139913188632384 tensor_quantizer.py:240] Call .cuda() if running on GPU after loading calibrated amax.
features.0._input_quantizer             : TensorQuantizer(8bit narrow fake per-tensor amax=2.7537 calibrator=MaxCalibrator scale=1.0 quant)
W0810 20:56:24.370319 139913188632384 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([64, 1, 1, 1]).
```

After QAT is completed, you should see the checkpoint of QAT model in the `$pwd/vgg16_ckpts` directory. For eg: `$pwd/vgg16_ckpts/ckpt_epoch110.pth`

## Exporting

Use the exporter script to create a torchscript module you can compile with Torch-TensorRT

### For PTQ
```
python3 export_ckpt.py <path-to-checkpoint>
```

The checkpoint file should be from the original training and not quatization aware fine tuning. THe script should produce a file called `trained_vgg16.jit.pt`

### For QAT
To export a QAT  model, you can run

```
python export_qat.py <path-to-checkpoint>
```

Please expect to see some warnings as indicated above when using `pytorch_quantization` toolkit. You can ignore these warnings and proceed to export the model.

This should generate a torchscript file named `trained_vgg16_qat.jit.pt`. You can use python or C++ API of Torch-TensorRT to run this quantized model using TensorRT.

## Citations

```
Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.
Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
```

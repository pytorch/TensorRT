# VGG16 Trained on CIFAR10

This is a recipe to train a VGG network on CIFAR10 to use with the TRTorch PTQ example.

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

### Quantization

To perform quantization aware training, it is recommended that you finetune your model obtained from previous step with quantization layers.

```
python3 train_qat.py --lr 0.01 --batch-size 128 --drop-ratio 0.15 --ckpt-dir $(pwd)/vgg16_ckpts --start-from 100 --epochs 110
```

After QAT is completed, you should see the checkpoint of QAT model in the `$pwd/vgg16_ckpts` directory. For eg: `$pwd/vgg16_ckpts/ckpt_epoch110.pth`

## Exporting

Use the exporter script to create a torchscipt module you can compile with TRTorch

```
python3 export_ckpt.py <path-to-checkpoint>
```

It should produce a file called `trained_vgg16.jit.pt`

To export a QAT  model, you can run

```
python export_qat.py <path-to-checkpoint>
```

This should generate a torchscript file named `trained_vgg16_qat.jit.pt`. You can use python or C++ API of TRTorch to run this quantized model using TensorRT.

## Citations

```
Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.
Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
```

# VGG16 Trained on CIFAR10

This is a recipe to train a VGG network on CIFAR10 to use with the TRTorch PTQ example.

## Prequisites

```
pip3 install -r requirements.txt --user
```

## Training

The following recipe should get somewhere between 89-92% accuracy on the CIFAR10 testset
```
python3 main.py --lr 0.01 --batch-size 128 --drop-ratio 0.15 --ckpt-dir $(pwd)/vgg16_ckpts --epochs 100
```

> 545 was the seed used in testing

You can monitor training with tensorboard, logs are stored by default at `/tmp/vgg16_logs`

## Exporting

Use the exporter script to create a torchscipt module you can compile with TRTorch

```
python3 export_ckpt.py <path-to-checkpoint>
```

It should produce a file called `trained_vgg16.jit.pt`

Once the trained VGG network is exported run it with the PTQ example.

## Citations

```
Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.
Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
```

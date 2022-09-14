.. _notebooks:

Example notebooks
===================

There exists a number of notebooks which cover specific using specific features and models
with Torch-TensorRT

Notebooks
------------

Compiling CitriNet with Torch-TensorRT
********************************************

Citrinet is an acoustic model used for the speech to text recognition task. It is a version
of QuartzNet that extends ContextNet, utilizing subword encoding (via Word Piece tokenization)
and Squeeze-and-Excitation(SE) mechanism and are therefore smaller than QuartzNet models. CitriNet
models take in audio segments and transcribe them to letter, byte pair, or word piece sequences.

This notebook demonstrates the steps for optimizing a pretrained CitriNet model with Torch-TensorRT,
and running it to test the speedup obtained.

* `Torch-TensorRT Getting Started - CitriNet <https://github.com/pytorch/TensorRT/blob/master/notebooks/CitriNet-example.ipynb>`_


Compiling EfficentNet with Torch-TensorRT
********************************************

EfficentNet is a feedforward CNN designed to achieve better performance and accuracy than alternative architectures
by using a "scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient".

This notebook demonstrates the steps for optimizing a pretrained EfficentNet model with Torch-TensorRT,
and running it to test the speedup obtained.

* `Torch-TensorRT Getting Started - EfficientNet-B0 <https://github.com/pytorch/TensorRT/blob/master/notebooks/EfficientNet-example.ipynb>`_


Masked Language Modeling (MLM) with Hugging Face BERT Transformer accelerated by Torch-TensorRT
*************************************************************************************************

"BERT is a transformer model pretrained on a large corpus of English data in a self-supervised fashion.
This way, the model learns an inner representation of the English language that can then be used to extract
features useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train
a standard classifier using the features produced by the BERT model as inputs." (https://huggingface.co/bert-base-uncased)

This notebook demonstrates the steps for optimizing a pretrained EfficentNet model with Torch-TensorRT,
and running it to test the speedup obtained.

* `Masked Language Modeling (MLM) with Hugging Face BERT Transformer <https://github.com/pytorch/TensorRT/blob/master/notebooks/Hugging-Face-BERT.ipynb>`_


Serving a model in C++ using Torch-TensorRT
**********************************************

This example shows how you can load a pretrained ResNet-50 model, convert it to a Torch-TensorRT
optimized model (via the Torch-TensorRT Python API), save the model as a torchscript module, and
then finally load and serve the model with the PyTorch C++ API.

* `ResNet C++ Serving Example <https://github.com/pytorch/TensorRT/blob/master/notebooks/Resnet50-CPP.ipynb>`_


Compiling ResNet50 with Torch-TensorRT
********************************************

This notebook demonstrates the steps for compiling a TorchScript module with Torch-TensorRT on a
pretrained ResNet-50 network, and running it to test the speedup obtained.

* `Torch-TensorRT Getting Started - ResNet 50 <https://github.com/pytorch/TensorRT/blob/master/notebooks/Resnet50-example.ipynb>`_


Using Dynamic Shapes with Torch-TensorRT
********************************************

Making use of Dynamic Shaped Tensors in Torch TensorRT is quite simple. Let's say you are
using the ``torch_tensorrt.compile(...)`` function  to compile a torchscript module. One
of the args in this function in this function is ``input``: which defines an input to a
module in terms of expected shape, data type and tensor format: ``torch_tensorrt.Input.``

For the purposes of this walkthrough we just need three kwargs: `min_shape`, `opt_shape`` and `max_shape`.

.. code-block:: py

    torch_tensorrt.Input(
            min_shape=(1, 224, 224, 3),
            opt_shape=(1, 512, 512, 3),
            max_shape=(1, 1024, 1024, 3),
            dtype=torch.int32
            format=torch.channel_last
        )
    ...

In this example, we are going to use a simple ResNet model to demonstrate the use of the API.

* `Torch-TensorRT - Using Dynamic Shapes <https://github.com/pytorch/TensorRT/blob/master/notebooks/dynamic-shapes.ipynb>`_

Using the FX Frontend with Torch-TensorRT
********************************************

The purpose of this example is to demostrate the overall flow of lowering a PyTorch model to TensorRT
conveniently with using FX.

* `Using the FX Frontend with Torch-TensorRT <https://github.com/pytorch/TensorRT/blob/master/notebooks/getting_started_with_fx_path_lower_to_trt.ipynb>`_


Compiling a PyTorch model using FX Frontend with Torch-TensorRT
*******************************************************************

The purpose of this example is to demonstrate the overall flow of lowering a PyTorch
model to TensorRT via FX with existing FX based tooling

* `Compiling a PyTorch model using FX Frontend with Torch-TensorRT  <https://github.com/pytorch/TensorRT/blob/master/notebooks/getting_started_with_fx_path_module.ipynb>`_


Compiling LeNet with Torch-TensorRT
*******************************************************************

This notebook demonstrates the steps for compiling a TorchScript module with Torch-TensorRT on a simple LeNet network.

* `Torch-TensorRT Getting Started - LeNet  <https://github.com/pytorch/TensorRT/blob/master/notebooks/lenet-getting-started.ipynb>`_


Accelerate Deep Learning Models using Quantization in Torch-TensorRT
*******************************************************************

Model Quantization is a popular way of optimization which reduces the size of models thereby
accelerating inference, also opening up the possibilities of deployments on devices with lower
computation power such as Jetson. Simply put, quantization is a process of mapping input values
 from a larger set to output values in a smaller set. In this notebook, we illustrate the workflow
 that you can adopt while quantizing a deep learning model in Torch-TensorRT. The notebook takes
 you through an example of Mobilenetv2 for a classification task on a subset of Imagenet Dataset
 called Imagenette which has 10 classes.

* `Accelerate Deep Learning Models using Quantization in Torch-TensorRT <https://github.com/pytorch/TensorRT/blob/master/notebooks/qat-ptq-workflow.ipynb>`_


Object Detection with Torch-TensorRT (SSD)
*******************************************************************

This notebook demonstrates the steps for compiling a TorchScript module with Torch-TensorRT on a pretrained SSD network, and running it to test the speedup obtained.

* `Object Detection with Torch-TensorRT (SSD)  <https://github.com/pytorch/TensorRT/blob/master/notebooks/ssd-object-detection-demo.ipynb>`_


Deploying Quantization Aware Trained models in INT8 using Torch-TensorRT
*****************************************************************************

Quantization Aware training (QAT) simulates quantization during training by
quantizing weights and activation layers. This will help to reduce the loss in
accuracy when we convert the network trained in FP32 to INT8 for faster inference.
QAT introduces additional nodes in the graph which will be used to learn the dynamic
ranges of weights and activation layers. In this notebook, we illustrate the following
steps from training to inference of a QAT model in Torch-TensorRT.

* `Deploying Quantization Aware Trained models in INT8 using Torch-TensorRT  <https://github.com/pytorch/TensorRT/blob/master/notebooks/vgg-qat.ipynb>`_

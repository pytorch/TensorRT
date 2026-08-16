.. _creating_a_ts_mod:

Creating a TorchScript Module
------------------------------
TorchScript is a way to create serializable and optimizable models from PyTorch code.
PyTorch has detailed documentation on how to do this https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html but briefly here is the
here is key background information and the process:

PyTorch programs are based around ``Module`` s which can be used to compose higher level modules. ``Modules`` contain a constructor to set up the modules, parameters and sub-modules
and a forward function which describes how to use the parameters and submodules when the module is invoked.

For example, we can define a LeNet module like this:

.. code-block:: python
    :linenos:

    import torch.nn as nn
    import torch.nn.functional as F


    class LeNetFeatExtractor(nn.Module):
        def __init__(self):
            super(LeNetFeatExtractor, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 3)
            self.conv2 = nn.Conv2d(6, 16, 3)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            return x


    class LeNetClassifier(nn.Module):
        def __init__(self):
            super(LeNetClassifier, self).__init__()
            self.fc1 = nn.Linear(16 * 6 * 6, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            self.feat = LeNetFeatExtractor()
            self.classifer = LeNetClassifier()

        def forward(self, x):
            x = self.feat(x)
            x = self.classifer(x)
            return x

.

    Obviously you may want to consolidate such a simple model into a single module but we can see the composability of PyTorch here

From here are two pathways for going from PyTorch Python code to TorchScript code: Tracing and Scripting.

Tracing follows the path of execution when the module is called and records what happens.
To trace an instance of our LeNet module, we can call ``torch.jit.trace`` with an example input.

.. code-block:: python

    import torch

    model = LeNet()
    input_data = torch.empty([1, 1, 32, 32])
    traced_model = torch.jit.trace(model, input_data)

Scripting actually inspects your code with a compiler and generates an equivalent TorchScript program. The difference is that since tracing
is following the execution of your module, it cannot pick up control flow for instance. By working from the Python code, the compiler can
include these components. We can run the script compiler on our LeNet module by calling ``torch.jit.script``

.. code-block:: python

    import torch

    model = LeNet()
    script_model = torch.jit.script(model)

There are reasons to use one path or another, the PyTorch documentation has information on how to choose. From a Torch-TensorRT prespective, there is
better support (i.e your module is more likely to compile) for traced modules because it doesn't include all the complexities of a complete
programming language, though both paths supported.

After scripting or tracing your module, you are given back a TorchScript Module. This contains the code and parameters used to run the module stored
in a intermediate representation that Torch-TensorRT can consume.

Here is what the LeNet traced module IR looks like:

.. code-block:: none

    graph(%self.1 : __torch__.___torch_mangle_10.LeNet,
        %input.1 : Float(1, 1, 32, 32)):
        %129 : __torch__.___torch_mangle_9.LeNetClassifier = prim::GetAttr[name="classifer"](%self.1)
        %119 : __torch__.___torch_mangle_5.LeNetFeatExtractor = prim::GetAttr[name="feat"](%self.1)
        %137 : Tensor = prim::CallMethod[name="forward"](%119, %input.1)
        %138 : Tensor = prim::CallMethod[name="forward"](%129, %137)
        return (%138)

and the LeNet scripted module IR:

.. code-block:: none

    graph(%self : __torch__.LeNet,
        %x.1 : Tensor):
        %2 : __torch__.LeNetFeatExtractor = prim::GetAttr[name="feat"](%self)
        %x.3 : Tensor = prim::CallMethod[name="forward"](%2, %x.1) # x.py:38:12
        %5 : __torch__.LeNetClassifier = prim::GetAttr[name="classifer"](%self)
        %x.5 : Tensor = prim::CallMethod[name="forward"](%5, %x.3) # x.py:39:12
        return (%x.5)

You can see that the IR preserves the module structure we have in our python code.

.. _ts_in_py:

Working with TorchScript in Python
-----------------------------------

TorchScript Modules are run the same way you run normal PyTorch modules. You can run the forward pass using the
``forward`` method or just calling the module ``torch_scirpt_module(in_tensor)`` The JIT compiler will compile
and optimize the module on the fly and then returns the results.

Saving TorchScript Module to Disk
-----------------------------------

For either traced or scripted modules, you can save the module to disk with the following command

.. code-block:: python

    import torch

    model = LeNet()
    script_model = torch.jit.script(model)
    script_model.save("lenet_scripted.ts")

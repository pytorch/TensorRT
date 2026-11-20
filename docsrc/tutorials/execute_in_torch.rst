.. _execute_in_torch:

Region annotations: PyTorch, custom operators, and kernels
==========================================================

A region is a group of operations that you place yourself. Torch-TensorRT
compiles the rest of the model as usual. This experimental API has three kinds
of region:

* ``execute_in_torch`` keeps the operations together in PyTorch, even if they
  have TensorRT converters. ``backend="eager"``, the default, runs them as
  PyTorch operations. ``backend="inductor"`` compiles them with PyTorch
  Inductor. CUDA tensors remain on CUDA inside the region.
* ``fused_op`` replaces the operations with an existing custom operator, which
  runs in PyTorch, outside TensorRT.
* ``fused_kernel`` replaces the operations with a kernel you supply, which runs
  inside TensorRT. Its plugin is registered automatically.

You can mark a region in two ways:

* Write a scope in ``forward``, such as
  ``with torch_tensorrt.region.execute_in_torch():``. The scope marks exactly
  the operations inside it.
* Pass rules when you compile, such as
  ``regions={"layers.*.mlp": torch_tensorrt.region.ExecuteInTorch()}``. A rule
  selects whole submodules by path, by class, or by the module object. The
  model's source stays as it is, so rules also work for a model from a library
  and for a program that is already exported. See :ref:`region_rules`.

This tutorial requires a CUDA device and compatible PyTorch and Torch-TensorRT
installations. Regions use private PyTorch APIs. If the installed PyTorch
lacks them, every trace that records a region raises an error. Its message
names the missing features after the words
``Regions (execute_in_torch, fused_op, fused_kernel and regions= rules) require``.
:ref:`region_ignore` shows how to trace a model without its regions.

Four complete examples are available:

* :ref:`execute_in_torch_example` runs the same region with eager and Inductor.
* :ref:`fused_op_example` replaces the region with a PyTorch custom operator.
* :ref:`fused_kernel_example` replaces the region with a supplied Triton kernel.
* :ref:`region_rules_example` selects submodules with rules on three compile
  paths.

Run them from the repository root:

.. code-block:: bash

    python examples/dynamo/execute_in_torch.py
    python examples/dynamo/fused_op.py
    python examples/dynamo/fused_kernel.py
    python examples/dynamo/region_rules.py

The eager/Inductor example requires Triton support for CUDA Inductor. The
custom-operator and rules examples require neither Triton nor QDP. The plugin
example requires Triton >=3.5 and TensorRT QDP support; see
:ref:`the kernel API prerequisites <torch_tensorrt_kernels_py>`.

Mark a region
-----------------

This model computes ``relu(x + 1) * 2``. Only the ReLU operation is marked to run
in PyTorch. Addition and multiplication remain eligible for TensorRT.

.. code-block:: python

    import torch
    import torch_tensorrt


    class Model(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            with torch_tensorrt.region.execute_in_torch(name="activation"):
                x = torch.relu(x)
            return x * 2


    model = Model().eval().cuda()
    x = torch.randn(4, 16, device="cuda")

The optional ``name`` labels diagnostics. Two scopes with the same name are
separate occurrences; leaving out ``name`` is also supported. During ordinary
eager execution the scope does nothing: its body runs as ordinary code.

Compile and check the result
--------------------------------

Pass the model to ``torch_tensorrt.compile`` with ``ir="dynamo"``:

.. code-block:: python

    compiled = torch_tensorrt.compile(
        model,
        inputs=[x],
        ir="dynamo",
        min_block_size=1,
    )

    with torch.no_grad():
        expected = model(x)
        actual = compiled(x)
        torch.testing.assert_close(actual, expected)

The trace records the region with the default ``strict=False`` and with
``strict=True``. Leave ``offload_module_to_cpu`` at its default, ``False``:
regions do not support CPU weight offload.

``min_block_size=1`` allows the single operations on either side of this small
example to become TensorRT engines. Placement outside the scope still depends
on converter support and the minimum block size in your application. Both the
fast and global partitioners preserve the region.

The resulting execution order is::

    TensorRT addition -> PyTorch ReLU -> TensorRT multiplication

.. _region_compile_paths:

Which compile paths honor regions
-------------------------------------

A scope written in ``forward`` is recorded whenever ``torch.export`` traces the
model, and whenever ``torch.compile`` traces it for the Torch-TensorRT backend.
You do not need to set anything up first. The same compile calls accept
``regions=`` rules.

.. list-table::
   :header-rows: 1
   :widths: 44 32 24

   * - How you compile
     - Scopes written in ``forward``
     - ``regions=`` rules
   * - ``torch_tensorrt.compile(model, ir="dynamo")``, with ``strict=True`` or
       the default ``strict=False``
     - Honored
     - Accepted
   * - Your own ``torch.export.export`` or ``torch_tensorrt.dynamo.trace``,
       then ``torch_tensorrt.dynamo.compile(program)`` or
       ``torch_tensorrt.compile(program, ir="dynamo")``
     - Honored
     - Accepted. Select by path or by class.
   * - A program saved with ``torch.export.save``, then loaded and compiled in
       another process
     - ``execute_in_torch``: honored. ``fused_op``: honored once that process
       has registered the operator. ``fused_kernel``: error, because the
       kernel is not saved.
     - Accepted, ``FusedKernel`` included. Select by path or by class.
   * - ``torch.compile(model, backend="torch_tensorrt")``, or
       ``backend="tensorrt"``
     - Honored. Pass ``dynamic=False``.
     - Accepted as ``options={"regions": ...}``
   * - ``torch_tensorrt.compile(model, ir="torch_compile")``
     - Honored
     - Accepted
   * - ``torch.compile`` with any other backend, and ``torch.jit.trace``
     - Ignored, with one warning
     - Not available
   * - ``MutableTorchTensorRTModule``
     - Error
     - Error

Where scopes are ignored, the body runs as ordinary code, and the operator or
kernel given to ``fused_op`` or ``fused_kernel`` is not used. Torch-TensorRT
logs one warning per process for other ``torch.compile`` backends and one for
TorchScript tracing. Ordinary eager execution also runs the body as ordinary
code, without a warning.

Pass rules to the compile call, not to the trace:
``torch_tensorrt.dynamo.trace(model, regions=...)`` raises a ``ValueError``.

``MutableTorchTensorRTModule`` raises a ``RegionError``: it rebuilds weights in
place, which regions do not support. Compile with ``torch_tensorrt.compile``
instead. For a model with scopes in ``forward``, you can also make the first
call of the module inside ``torch_tensorrt.region.ignore_regions()``; the
scopes are then compiled as ordinary operations (see :ref:`region_ignore`).
Rules cannot be used with it.

:ref:`region_export_later` and :ref:`region_torch_compile` describe the paths
after the first row. :ref:`region_rules` describes the rules.

Compile a region with Inductor
---------------------------------

Inside ``forward``, set ``backend="inductor"`` on the scope and use the same
compile call as above:

.. code-block:: python

    with torch_tensorrt.region.execute_in_torch(backend="inductor"):
        x = torch.relu(x)

The surrounding operations remain eligible for TensorRT. The first call to
the compiled model compiles the region with Inductor; later calls reuse it.
See :ref:`execute_in_torch_example` for a complete model and execution checks.
A rule makes the same choice with ``ExecuteInTorch(backend="inductor")``.

Inductor compiles the whole region with static shapes. It may fuse compatible
operations, but a region does not guarantee one GPU kernel. Compilation errors
are reported to the caller instead of silently falling back to eager execution.

Inductor compilation is lazy: completing ``torch_tensorrt.compile`` does not mean
the region has already been compiled, and a ``copy.deepcopy`` of the compiled
model compiles the region again on the copy's first call. Keep the same inference
context for warmup and measured calls, and exclude the first call from
steady-state timings. Changes to input layout or execution context can cause
recompilation even when shapes are unchanged. Dry runs retain the reference child
and do not install an Inductor callable.

Only ``"eager"`` and ``"inductor"`` are accepted. ``backend`` is the compiler
choice, not the ``mode`` argument of ``torch.compile``: values such as
``"reduce-overhead"`` and ``"max-autotune"`` are not valid here. The region API
does not expose compiler tuning options. Its Inductor configuration disables
``triton.cudagraphs``. For whole-model CUDA graphs, use
``torch_tensorrt.runtime.enable_cudagraphs(compiled)``. For either eager
or Inductor regions, the runtime warms up the model before capture in the
caller's execution context. Under ``torch.autocast``, warmup and capture bypass
autocast's weight cache, as with ``cache_enabled=False``: the graph records the
casts, so each replay casts the current weights.

CUDA graph capture requires strided tensor inputs on one CUDA device. It copies
the bytes each input reaches, from its first element to its last, and keeps its
strides; inputs whose bytes overlap, such as the results of ``chunk``, share one
copy. The CUDA graphs tutorial lists
:ref:`what triggers capture again <cuda_graphs_recording>`. As with
``torch.compile``, a region that reads outside its input views, for example with
``as_strided`` and an explicit ``storage_offset``, can fail or read different
values under CUDA graphs. The copies keep the memory alignment that the inputs
had at capture, so kernels that pick a vectorized path by alignment, such as
reductions, can make replays with differently aligned inputs, such as other
slices of a larger tensor, differ from eager execution in the last bits.
Warmup preserves the default PyTorch CPU and CUDA RNG states, including on recapture.
Capture cannot copy unpinned CPU tensors to the GPU. Eager regions copy CPU
scalars passed to operations such as ``torch.where``, and TensorRT engines copy
CPU scalars that regions return; create such constants on the input's device.

As with the default backend, this context manager does nothing during ordinary
eager execution. It enables Inductor only through Torch-TensorRT compilation.
The ``name`` keyword remains optional and can be combined with either backend.

Replace a region with a PyTorch custom operator
-----------------------------------------------

Use ``fused_op`` when an existing custom operator implements the region and
should execute through PyTorch. Register the operator and its fake implementation
before compiling. The fake implementation describes the output without running
the real implementation during capture.

.. code-block:: python

    @torch.library.custom_op("region_tutorial::bias_relu_scale", mutates_args=())
    def bias_relu_scale(
        x: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return torch.relu(x + bias) * scale


    @bias_relu_scale.register_fake
    def bias_relu_scale_fake(x, bias, scale):
        return torch.empty_like(x)


    class CustomOpModel(torch.nn.Module):
        def forward(self, x, bias, scale):
            x = x + 1.0
            with torch_tensorrt.region.fused_op(
                torch.ops.region_tutorial.bias_relu_scale.default,
                inputs=(x, bias, scale),
                name="bias_relu_scale",
            ):
                x = torch.relu(x + bias) * scale
            return x - 0.5

Compile an eval-mode instance with three equal-shaped CUDA inputs using the
compile call above. The execution order is::

    TensorRT addition -> PyTorch custom operator -> TensorRT subtraction

The body defines the reference calculation and runs during ordinary eager
execution. Compilation replaces the entire body with the supplied operator;
it does not generate a new implementation or a TensorRT plugin. See
:ref:`fused_op_example` for a complete run that checks outputs, placement, and
actual invocation of the custom operator.

Passing the decorated ``bias_relu_scale`` function, or
``torch.ops.region_tutorial.bias_relu_scale`` without ``.default``, selects
the same operator.

The operator must be a custom operator; a built-in PyTorch operator is
rejected. Its schema must accept positional Tensor arguments and return
tensors, with no mutation or aliasing. A scope checks the operator when the
model is traced. A ``FusedOp`` rule checks it when you create the rule.

``inputs`` binds the operator's arguments: list every tensor the region reads,
including parameters and buffers, in the operator's argument order. Here the
first argument is the result of ``x + 1``, followed by ``bias`` and ``scale``.
Compilation cannot infer this order from the body: in ``up * F.silu(gate)``,
``silu`` reads ``gate`` before the multiplication reads ``up``. ``inputs`` may
be omitted only when the region reads at most one tensor, and compilation
rejects a list that omits, repeats, or adds a tensor. A tensor literal such as
``torch.tensor(2.0)`` is lifted out of the region as an input, so create it
before the region and list it. Outputs follow their production order in the
region. The registered fake implementation must match the captured output
metadata. These boundaries currently have static shapes. The caller must verify
that the actual operator implements the complete reference calculation.

``fused_op`` also permits random operations in the reference body, for custom
operators that own their RNG behavior. Numerical tests must account for the
operator's random-state contract. ``execute_in_torch`` and ``fused_kernel``
continue to reject random regions.

A traced program records the operator by name, here
``region_tutorial::bias_relu_scale.default``. The process that compiles the
program looks the operator up by that name, so it must have registered the
operator first. If it has not, compilation raises a ``RegionError`` that names
the missing operator.

Register the custom operator's schema and implementation before loading a saved
model in a new process. Saving to ExecuTorch (see `Save the compiled model`_)
keeps the custom operator in the ``.pte`` as its ``out`` overload, so define
one; without it, ExecuTorch fails with ``Missing out variants``:

.. code-block:: python

    torch.library.define(
        "region_tutorial::bias_relu_scale.out",
        "(Tensor x, Tensor bias, Tensor scale, *, Tensor(a!) out) -> Tensor(a!)",
    )


    @torch.library.impl(
        "region_tutorial::bias_relu_scale.out", "CompositeExplicitAutograd"
    )
    def bias_relu_scale_out(x, bias, scale, *, out):
        return out.copy_(torch.relu(x + bias) * scale)

The application's ExecuTorch runtime must register a kernel for that overload.

Replace a region with a custom-kernel plugin
----------------------------------------------

Use ``fused_kernel`` when you already have a kernel that implements the entire
annotated computation. For the Triton kernel and launch callbacks defined in
:ref:`fused_kernel_example`, the annotation is:

.. code-block:: python

    with torch_tensorrt.region.fused_kernel(
        kernel=lambda: bias_relu_scale_kernel,
        inputs=(x, bias, scale),
        constexprs={"BLOCK": BLOCK},
        grid=launch_grid,
        extra_args_fn=kernel_extras,
    ):
        x = x + bias
        x = torch.relu(x)
        x = x * scale

``kernel`` must be a zero-argument function returning an existing ``@triton.jit``
kernel or a ``torch_tensorrt.kernels.KernelSpec`` for CUDA C++; passing the
kernel itself is rejected, even in eager execution. Strict export
(``strict=True``) and ``torch.compile`` call the function outside the traced
tensor program. On those paths a lambda written in ``forward`` must refer to a
global kernel, not to ``self`` or a local variable. To use a kernel chosen in
``__init__``, create the factory there, for example
``self.kernel_factory = lambda: kernel``, and pass
``kernel=self.kernel_factory``. Both forms work on every compile path.
``grid`` and ``extra_args_fn`` follow the same rules: all three must be
functions or ``functools.partial`` objects, not methods. ``name`` is optional.

Supply the kernel and its launch configuration. Torch-TensorRT uses the
annotated operations to determine the kernel's inputs and outputs, then
registers the operator and TensorRT plugin automatically. You do not write
registration functions or an FX rewrite. The selected operations must become
part of a TensorRT engine; compilation reports an error if that is impossible.
A dry run logs that error as a warning and names the region in its report.
Calling the original model eagerly still executes the PyTorch body.

The kernel is a Python object. A traced program does not hold it: the program
holds a key that finds the kernel only in the process that traced the program.
Compile an exported program with a ``fused_kernel`` scope in that same process.
In any other process, compilation raises a ``RegionError``. To use a kernel
with a program that another process exported, select the module with a
``FusedKernel`` rule when you compile (see :ref:`region_rules`).

Each compilation registers a new operator and plugin for every ``fused_kernel``
region. They remain for the lifetime of the Python process, also when the
compilation fails after registration, so compiling again adds more; cleanup is
not implemented. The prepared kernel itself is kept once: asking again for the
same kernel with the same options reuses it.

For Triton, runtime kernel parameters must be ordered as input pointers,
optional ``i32`` scalar extras, then output pointers. Input pointers follow
``inputs``, which lists tensors as it does for ``fused_op``, including
parameters and buffers. Outputs are the tensors used after the scope, in the
order they were produced inside it. This can differ from the model's final
return order: if ``a`` is computed before ``b`` and both leave the region, the
output pointers are for ``a``, then ``b``, even if the model returns ``(b, a)``.
Pointer dtypes are inferred from capture. Supply ``constexprs`` and the
required ``grid(inputs, outputs)`` callback, plus
``extra_args_fn(inputs, outputs)`` when scalar launch arguments are needed.
Those callbacks receive TensorRT tensor descriptors. A CUDA ``KernelSpec``
provides its own launch configuration.

Plugin boundaries currently require static, contiguous CUDA tensors with the
captured shapes and dtypes. Inputs and outputs must satisfy the rules in
:ref:`region_body`. The kernel author is responsible for numerical equivalence
to the entire body: automatic registration does not generate the kernel or
prove its correctness. Compare the compiled result against the original model
on representative inputs, as the runnable example does.

Use ``min_block_size=1`` for a small example. ``require_full_compilation=True``
is allowed when the model has plugin regions and no ``execute_in_torch`` or
``fused_op`` regions. The example compiles its surrounding add and subtract plus
the custom kernel into one TensorRT engine.

Select an invocation of a shared module
-------------------------------------------

A scope applies only to the operations captured inside it. For example, if
``self.block`` is called three times, the following marks only its second call:

.. code-block:: python

    class SharedBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU())

        def forward(self, x):
            x = self.block(x)
            with torch_tensorrt.region.execute_in_torch():
                x = self.block(x)
            return self.block(x)

A rule cannot do this: it selects every call of the module, and each call
becomes its own region.

.. _region_body:

Write the body of a region
------------------------------

The body of a region is the code inside a scope, or the ``forward`` of a
module that a rule selects. It must be tensor code that the trace can record.
A region does not restore Python hooks or side effects that the trace drops.
Reassigning a Python variable, such as ``x = torch.relu(x)``, is allowed.

**Inputs.** A region can read parameters, buffers, and noncontiguous tensors.
Tensor constants created with ``torch.tensor`` inside an ``execute_in_torch``
region are supported. Region inputs may share storage, such as the results of
``chunk`` or ``unbind``, a tensor and its transpose, or the weights of a CUDA
``nn.LSTM``, which cuDNN flattens into one buffer. ``execute_in_torch`` regions
read these inputs without copying them. ``fused_op`` and ``fused_kernel``
regions receive a copy of each input that shares storage with an earlier one:
parameters and buffers are copied once during compilation, and other inputs
may be copied on every call.

**Casts.** Dtype casts such as ``.float()`` or ``.to(torch.float16)`` are
supported inside a region and on its outputs, for example to compute a
normalization in fp32 within an fp16 model.

**Results.** A region can return several tensors. They must not share storage
with the region's inputs or with each other:

* In a scope written in ``forward``, clone a view or a ``chunk`` result before
  it leaves the scope. Without the clone, compilation raises a
  ``RegionCaptureError`` that asks you to clone the outputs inside the region.
* A rule selects a module that you cannot edit, so compilation makes the copy
  itself. A selected module may return a view of its input, such as a
  transpose or the results of ``chunk``.

**In-place operations.** An in-place operation is accepted when it writes a
tensor that the region itself created and that nothing else reads.
Torch-TensorRT makes the operation out of place. Examples are
``nn.ReLU(inplace=True)`` after a linear layer inside a selected module, and
``y = x * 3`` followed by ``y.relu_()`` in a scope. A write to an input of the
region, or to a tensor that a view made before the write also reads, raises a
``RegionCaptureError`` that starts with
``Mutation or effects inside regions are unsupported``.

**Operations that return their input.** Eval-mode dropout and a cast to the
dtype a tensor already has return their input itself.

* An ``execute_in_torch`` region may contain them.
* In a ``fused_op`` or ``fused_kernel`` scope, such an operation makes the
  input an output of the scope when code after the scope reads the input or
  the result. Compilation raises an error; clone the input inside the scope
  before that operation.
* With a ``FusedOp`` or ``FusedKernel`` rule, a tensor that the module hands
  through unchanged stays the caller's tensor. The operator or kernel does not
  produce it. A module that only hands its input through, such as an eval-mode
  ``nn.Dropout``, leaves the operator or kernel nothing to do, and compilation
  raises a ``RegionCaptureError``. Select a module that computes something.

**Random operations.** Operations that draw random numbers, such as
``torch.rand_like`` or dropout with ``training=True``, are supported only in
``fused_op`` regions. Eval-mode dropout and attention with ``dropout_p=0``
draw none and are allowed in every region.

.. _region_rules:

Select submodules with rules
--------------------------------

A rule places every call of a submodule in a region when you compile. You do
not edit the model. Use rules for a model whose source you cannot change, such
as a model from a library, and for a program that is already exported.

This model has no region code. The custom operator computes what ``Norm``
computes:

.. code-block:: python

    @torch.library.custom_op("region_tutorial::scale_norm", mutates_args=())
    def scale_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight


    @scale_norm.register_fake
    def scale_norm_fake(x, weight):
        return torch.empty_like(x)


    class Norm(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(width) + 0.5)

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


    class Mlp(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.up = torch.nn.Linear(width, 2 * width)
            self.down = torch.nn.Linear(2 * width, width)

        def forward(self, x):
            return self.down(torch.nn.functional.silu(self.up(x)))


    class Block(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.norm = Norm(width)
            self.mlp = Mlp(width)

        def forward(self, x):
            return x + self.mlp(self.norm(x))


    class Layers(torch.nn.Module):
        def __init__(self, width=16, depth=2):
            super().__init__()
            self.embed = torch.nn.Linear(width, width)
            self.layers = torch.nn.ModuleList(Block(width) for _ in range(depth))
            self.head = torch.nn.Linear(width, 4)

        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            return self.head(x)


    layers = Layers().eval().cuda()
    x = torch.randn(4, 16, device="cuda")

Pass the rules as ``regions={selector: rule}``:

.. code-block:: python

    from torch_tensorrt.region import ExecuteInTorch, FusedOp

    rules = {
        "layers.*.mlp": ExecuteInTorch(),
        Norm: FusedOp(scale_norm, inputs=[0, "weight"]),
    }
    compiled = torch_tensorrt.compile(
        layers,
        inputs=[x],
        ir="dynamo",
        min_block_size=1,
        regions=rules,
    )

    with torch.no_grad():
        # TensorRT runs linear layers in TF32 by default: allow a small difference.
        torch.testing.assert_close(compiled(x), layers(x), rtol=1e-2, atol=1e-2)

Each call of a selected module becomes one region. Here there are four:
``layers.0.norm`` and ``layers.1.norm`` call the operator, and ``layers.0.mlp``
and ``layers.1.mlp`` run in PyTorch. The model must be in eval mode.

Rules work the same way for a model from a library, such as a Hugging Face
Transformers model. Use the paths that ``model.named_modules()`` prints, such
as ``"model.layers.*.mlp"``.

Selectors
^^^^^^^^^

A selector is a module path, a module class, or a module object.

* A path is a name that ``model.named_modules()`` prints, such as
  ``"layers.0.mlp"``.
* ``*`` stands for one name: ``"layers.*.mlp"`` selects ``layers.0.mlp`` and
  ``layers.1.mlp``. It does not cross a dot, so ``"*.mlp"`` selects nothing in
  this model.
* ``**`` stands for any number of names, including none: ``"**.mlp"`` selects
  every module named ``mlp`` at any depth.
* Each name is matched as a shell pattern, so ``"layers.[01].ml*"`` also
  selects both ``mlp`` modules.
* A class selects every module of that class and of its subclasses, such as
  ``Norm`` or ``torch.nn.Linear``. The model's own class does not select the
  whole model.
* A module object selects that module. Use it when you compile the
  ``torch.nn.Module``, ahead of time or with ``torch.compile``:

.. code-block:: python

    compiled = torch_tensorrt.compile(
        layers,
        inputs=[x],
        ir="dynamo",
        min_block_size=1,
        regions={layers.layers[0].mlp: ExecuteInTorch()},
    )

An exported program holds no module objects. It records each module by its
path and by the name of its class:

* A module object as a selector raises a ``RegionCaptureError``. Select by
  path or by class.
* A class selects subclasses too when the recorded class is loaded in the
  compiling process, that is, when the Python module that defines it is
  already imported. Torch-TensorRT does not import it for you. Otherwise the
  class selects only the modules whose recorded class has the same name and
  comes from the same Python module. A path works everywhere.

A module registered under two names, as in ``self.second = self.first``, is
one module. When you compile the ``torch.nn.Module``, selecting it under
either name selects every call of it. An exported program records only names:
the default export records each call under the name it was called by, and a
``strict=True`` export records every call under the first name. Select the
names that the program recorded; the error for a name it did not record lists
them.

A selected module includes its children: ``"layers.*"`` puts each whole block
in one region. Regions cannot nest or overlap, so these raise a
``RegionCaptureError``:

* two rules that select the same operations, such as ``"layers.0"`` and
  ``"layers.0.mlp"``;
* a rule that selects a module whose ``forward`` already holds a scope.

A rule that matches no module also raises a ``RegionCaptureError``. Its
message lists some of the model's module paths. A container such as a
``ModuleList`` is never called, so a rule for ``"layers"`` places nothing in a
region: select its children with ``"layers.*"``. The error for such a rule
says so. Rules and scopes written in ``forward`` can be used in the same model
as long as they do not overlap.

``regions`` must be a mapping. Any other value, and a selector that is not a
path, a class, or a module, raises a ``TypeError``.

The three rules
^^^^^^^^^^^^^^^

Each rule is the compile-time form of one scope:

* ``ExecuteInTorch(backend="eager", name=None)`` keeps the module in PyTorch.
* ``FusedOp(op, inputs=None, name=None)`` replaces the module with a custom
  operator.
* ``FusedKernel(kernel, inputs=None, name=None, ...)`` replaces the module with
  a kernel inside TensorRT. It takes the options of ``fused_kernel``:
  ``signature``, ``constexprs``, ``grid``, ``extra_args_fn``, ``num_warps``,
  and ``num_stages``.

``name`` labels diagnostics. It defaults to the module's path. The
requirements of each scope apply to its rule: the operator needs a fake
implementation, the kernel must implement the whole module, and so on.

List the inputs of an operator or kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``inputs`` lists the arguments of the operator or kernel, in order. A scope
lists tensors. A rule has no tensors to list, so it lists positions and names:

* An integer is a position among the tensors that the module call receives.
  Positional arguments come first, then keyword arguments in the order the
  call passes them. Arguments that are not tensors are not counted.
* A string names a parameter or buffer of the module, such as ``"weight"``, or
  ``"proj.weight"`` for a parameter of its child ``proj``. A weight that the
  module shares with another module, such as a tied output layer, is named
  the same way.

``inputs=[0, "weight"]`` above passes the tensor that ``Norm`` receives, then
its ``weight``. List every tensor the module reads. Omit ``inputs`` only when
the module reads one tensor. If the list is incomplete, the error names what
to add, for example ``add 'shift'`` or ``position 0, 'weight', 'shift'``.

A tensor constant that the module creates in ``forward`` with ``torch.tensor``
also counts as a tensor the module reads. The trace gives it a generated name,
such as ``lifted_tensor_0``, which the error shows. An ``ExecuteInTorch`` rule
needs no ``inputs`` and takes such a module as it is.

A call may pass one tensor at two positions, as in ``self.mix(y, y)``. List
both positions, ``inputs=[0, 1]``: the operator or kernel receives the tensor
twice.

A position needs the order in which the call passes its tensors:

* ``torch_tensorrt.compile(model, ir="dynamo", regions=...)`` records that
  order when it traces the module. Any position works. The exception is a
  module that is called with different argument layouts, such as
  ``self.mix(a, b)`` and ``self.mix(a, b=c)``: ``torch.export`` cannot record
  its calls, and Torch-TensorRT logs a warning that says so.
* A program you export yourself records it for the modules you name in
  ``preserve_module_call_signature``.
* Otherwise the order is unknown. That is the case for a program exported
  without that argument, and for ``torch.compile``. Position ``0`` still works
  when the module receives a single tensor. When it receives more, compilation
  raises a ``RegionCaptureError`` instead of guessing. The message names the
  tensors and says how to make the order known.

This module receives two tensors, so its exported program needs the call
signature:

.. code-block:: python

    @torch.library.custom_op("region_tutorial::mix", mutates_args=())
    def mix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a - 2 * b


    @mix.register_fake
    def mix_fake(a, b):
        return torch.empty_like(a)


    class Mix(torch.nn.Module):
        def forward(self, a, b):
            return a - 2 * b


    class TwoInputs(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mix = Mix()

        def forward(self, x):
            return self.mix(x + 1, b=x * 3) * 2


    two_inputs = TwoInputs().eval().cuda()
    program = torch.export.export(
        two_inputs, (x,), preserve_module_call_signature=("mix",)
    )
    compiled = torch_tensorrt.dynamo.compile(
        program,
        inputs=[x],
        min_block_size=1,
        regions={"mix": FusedOp(mix, inputs=[0, 1])},
    )

Return the results of the module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The operator or kernel of a ``FusedOp`` or ``FusedKernel`` rule returns what
the module returns, in the order the module returns it. This module computes
``low`` first and returns it second, so its operator returns ``(high, low)``:

.. code-block:: python

    @torch.library.custom_op("region_tutorial::split", mutates_args=())
    def split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x + 1, x - 1


    @split.register_fake
    def split_fake(x):
        return torch.empty_like(x), torch.empty_like(x)


    class Split(torch.nn.Module):
        def forward(self, x):
            low = x - 1
            high = x + 1
            return high, low


    class TwoResults(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.split = Split()

        def forward(self, x):
            high, low = self.split(x * 2)
            return high * 3 + low


    two_results = TwoResults().eval().cuda()
    compiled = torch_tensorrt.compile(
        two_results,
        inputs=[x],
        ir="dynamo",
        min_block_size=1,
        regions={"split": FusedOp(split)},
    )

The return order comes from the same record as the call order of the inputs:

* With the record, the operator or kernel returns every tensor the module
  returns, also one that the rest of the model never reads. A tensor that the
  module hands through unchanged is the exception: it is not a result of the
  operator or kernel (see :ref:`region_body`).
* Without the record, the operator or kernel returns the one tensor that the
  rest of the model reads. If the model reads more than one result of the
  module, compilation raises a ``RegionCaptureError`` that says the program
  does not record the order.

A program that is already exported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rules also select regions in a program that was exported without any region
code, in this process or in another one:

.. code-block:: python

    torch.export.save(torch.export.export(layers, (x,)), "layers.pt2")

    # Later, or in another process:
    program = torch.export.load("layers.pt2")
    compiled = torch_tensorrt.dynamo.compile(
        program,
        inputs=[x],
        min_block_size=1,
        regions={
            "layers.*.mlp": ExecuteInTorch(),
            "layers.*.norm": FusedOp(scale_norm, inputs=[0, "weight"]),
        },
    )

``torch_tensorrt.compile(program, ir="dynamo", regions=...)`` does the same.
The rule supplies the operator or kernel when you compile, so a ``FusedKernel``
rule works in a process that did not export the program.

Compilation works on a copy of the program's graph. Your ``program`` stays as
it was exported, and you can compile it again with other rules.

.. _region_torch_compile:

Use regions with torch.compile
----------------------------------

The Torch-TensorRT ``torch.compile`` backend honors scopes written in
``forward``. Pass ``dynamic=False``:

.. code-block:: python

    optimized = torch.compile(
        model,
        backend="torch_tensorrt",
        dynamic=False,
        options={"min_block_size": 1},
    )

    with torch.no_grad():
        torch.testing.assert_close(optimized(x), model(x))

Rules go in the backend options:

.. code-block:: python

    optimized = torch.compile(
        layers,
        backend="torch_tensorrt",
        dynamic=False,
        options={"min_block_size": 1, "regions": rules},
    )

The same call through ``torch_tensorrt.compile`` passes ``dynamic=False`` for
you:

.. code-block:: python

    optimized = torch_tensorrt.compile(
        layers,
        ir="torch_compile",
        min_block_size=1,
        regions=rules,
    )

An option name that the backend does not know, such as a misspelled
``"region"``, is ignored with a warning that names it.

When errors appear
^^^^^^^^^^^^^^^^^^

``torch.compile`` only wraps the model. It traces and compiles on the first
call, so that call is where an error in a scope or a rule appears:

* An error from the backend arrives as
  ``torch._dynamo.exc.BackendCompilerFailed``. Its message holds the region
  message, and its ``inner_exception`` is the ``RegionError`` or
  ``RegionCaptureError``.
* An error raised while a scope is traced arrives as
  ``torch._dynamo.exc.InternalTorchDynamoError``, with the region message in
  its text. For a scope, a PyTorch without the features that regions need is
  reported this way.
* A graph break inside a scope arrives as ``torch._dynamo.exc.Unsupported``.
* An invalid argument of a scope, such as an unknown ``backend``, raises its
  ``ValueError`` unchanged.

``torch_tensorrt.compile(model, ir="torch_compile", regions=...)`` checks the
rules when you call it. A rule that selects no submodule and a model in
training mode raise a ``RegionCaptureError`` there, before any call.

Nothing falls back silently. If a graph with regions cannot be compiled, the
call raises a ``RegionError`` that starts with
``Compilation of a torch.compile graph with regions failed`` and names the
cause. It does not run the model in PyTorch without the regions. A graph
without regions falls back to PyTorch as before.

Shapes
^^^^^^

Shapes must be static. With ``dynamic=False``, each new input shape is
compiled on its own. Without it, a call with a new shape makes
``torch.compile`` trace again with dynamic shapes, and that raises an error
that asks for ``dynamic=False``.

``torch.compile`` compiles a model for a limited number of input shapes:
``torch._dynamo.config.recompile_limit``, 8 by default. After that it logs
``hit config.recompile_limit`` and runs further shapes in PyTorch, without the
regions. This is PyTorch behavior, and Torch-TensorRT cannot detect it. Set
``torch._dynamo.config.fail_on_recompile_limit_hit = True`` to get an error
instead.

Graph breaks
^^^^^^^^^^^^

A graph break is a place where ``torch.compile`` cannot continue its trace,
for example a ``print`` call. ``torch.compile`` then traces the model as
several graphs.

* A graph break between scopes is fine. Each graph keeps its own scopes.
* A graph break inside a scope is an error. ``fullgraph=True`` makes
  ``torch.compile`` report every graph break instead of splitting the model.
* ``torch.compile`` does not trace a ``forward`` that has a graph break inside
  a loop. It runs that ``forward`` in Python and traces the functions it
  calls. A scope written directly in such a ``forward`` runs as ordinary code,
  and Torch-TensorRT logs one warning that says the annotations were ignored
  in a ``forward`` that ``torch.compile`` did not trace.

How rules find their modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rules are matched against the modules of the model, not against the graphs
that ``torch.compile`` traces:

* A rule applies in every graph that holds its module. If a graph break splits
  the model, a rule for a module in one graph is not an error in the others.
* A path starts at the model you compiled. ``"act"`` selects the model's own
  ``act``, never a module named ``act`` inside a submodule.
* ``torch_tensorrt.compile(model, ir="torch_compile", regions=...)`` matches
  the rules when you call it. ``torch.compile(model, ...)`` matches them on
  the first call; the backend finds the model from the running
  ``torch.compile`` call.
* If you compile a function instead of the model, there is no model for a
  path to start from, and a path raises a ``RegionCaptureError``. Select by
  class or by module object there.
* A rule needs the whole call of its module in one graph. If a graph break
  makes ``torch.compile`` trace part of the module's ``forward`` as a graph of
  its own, compiling the model raises a ``RegionCaptureError`` that names the
  module. Remove the graph break, or write the scope in the module's
  ``forward``.

``torch.compile`` does not record the order of a module call's tensors. With
``FusedOp`` and ``FusedKernel`` rules, a position in ``inputs`` works when the
module receives one tensor, and the module must return one tensor that the
model reads. Names work as usual. For other modules, compile ahead of time
with ``ir="dynamo"``, or write the scope in the module's ``forward``.

What the backend rejects
^^^^^^^^^^^^^^^^^^^^^^^^

These checks apply to a graph with a live region, that is, a region whose
results the model uses:

* **Training mode.** When you compile the model itself, a module in training
  mode raises ``Regions currently require model.eval()``. An eval-mode model
  may keep dropout active on purpose outside its regions. When you compile a
  function that calls the model, the backend cannot see the model's mode, so
  dropout that drops values outside a region is rejected.
* **Writes to inputs and buffers.** A write to a graph input or to a buffer,
  and batch norm in training mode with running statistics, raise an error that
  starts with ``Explicit regions require a functional model``.
* **Autocast and grad-mode blocks.** A region inside a ``torch.autocast``
  block, and such a block inside a region, raise a ``RegionCaptureError``. Use
  explicit casts such as ``.float()``, or enter the block around the call of
  the compiled model. The same holds for a ``torch.no_grad()`` or
  ``torch.enable_grad()`` block that changes the grad mode: enter it around
  the call. A ``torch.no_grad()`` block in a model that is called under
  ``torch.no_grad()`` changes nothing and is accepted.
* **In-place operations.** The rules of :ref:`region_body` apply. An in-place
  call on an input of the region, written with a trailing underscore, with
  ``inplace=True`` or with ``out=``, raises
  ``Mutation or effects inside regions are unsupported``.
* **Higher-order transforms.** A graph that uses ``torch.vmap`` is rejected.
* **Distributed tracing.** ``use_distributed_mode_trace=True`` is rejected.

The result cannot be saved or inspected. It is not a graph module, so
``torch_tensorrt.save`` rejects it. Compile with ``ir="dynamo"`` when you need
to save the model or look at its graph.

An ``execute_in_torch`` region with ``backend="inductor"`` is compiled by
Inductor once, on the first call, as it is with ``ir="dynamo"``.

.. _region_export_later:

Export first, compile later
-------------------------------

``torch.export.export`` records the scopes written in ``forward``, with
``strict=True`` or without. ``torch_tensorrt.dynamo.trace`` does too. Compile
the program whenever you want:

.. code-block:: python

    program = torch.export.export(model, (x,))
    compiled = torch_tensorrt.dynamo.compile(program, inputs=[x], min_block_size=1)

The program holds two marker operators around each scope,
``torch_tensorrt_region::begin`` and ``torch_tensorrt_region::end``. They do
nothing when the program runs in PyTorch. Compilation works on a copy of the
program's graph, so your program keeps its markers and can be compiled again.

To compile in another process, save the program with ``torch.export.save``.
In the other process:

* Import ``torch_tensorrt`` before ``torch.export.load``. The import registers
  the marker operators; without it, loading fails.
* An ``execute_in_torch`` scope needs nothing else.
* A ``fused_op`` scope needs its operator. Import the code that registers it
  before you compile.
* A ``fused_kernel`` scope cannot be compiled there, because the kernel is not
  saved. Compile it in the exporting process. In another process, select a
  submodule with a ``FusedKernel`` rule instead of writing the scope.

A loaded program is compiled like the freshly exported one. Its regions may
also read tensors that share storage, such as the results of ``chunk`` or the
weights of an ``nn.LSTM``.

Do not call ``program.run_decompositions()`` before you compile. It rewrites
the markers, and compilation then raises a ``RegionCaptureError``. Compile the
program as ``torch.export`` produced it. Torch-TensorRT runs its own
decompositions.

.. _region_ignore:

Export for another tool
---------------------------

The markers are custom operators of Torch-TensorRT. Another tool that reads
exported programs may not know them. For example,
``torch.onnx.export(model, (x,), dynamo=True)`` of the model above fails with a
``ConversionError`` that names ``torch_tensorrt_region.begin``. To export a
model with scopes for such a tool, trace it inside ``ignore_regions``:

.. code-block:: python

    with torch_tensorrt.region.ignore_regions():
        plain_program = torch.export.export(model, (x,))
        onnx_program = torch.onnx.export(model, (x,), dynamo=True)

Inside the block the scopes are traced as ordinary operations, so the program
holds no markers and the ONNX export succeeds. The block must be active while
the trace runs:

* ``torch.jit.trace`` inside the block also logs no warning about ignored
  scopes.
* ``torch.compile`` traces on the first call of the compiled model, so the
  block must surround that call, not the ``torch.compile`` call. The choice
  then stays with the compiled code: later calls reuse it, inside or outside
  the block. ``torch._dynamo.reset()`` discards the compiled code.
* ``MutableTorchTensorRTModule`` compiles on its first call. Make that call
  inside the block to use it with a model that has scopes.

The block does not affect ``regions=`` rules.

Save the compiled model
---------------------------

The legacy exported-program path supports ``execute_in_torch`` regions with
``backend="eager"``, ``fused_op`` regions, and ``fused_kernel`` regions already
built into TensorRT engines. It does not matter whether a region came from a
scope or from a rule. Save the already chosen placement as follows:

.. code-block:: python

    torch_tensorrt.save(
        compiled,
        "model.ep",
        output_format="exported_program",
        retrace=False,
        use_legacy_exporter=True,
    )
    restored = torch_tensorrt.load("model.ep").module()

    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x))

Serialization works on a copy of the graph, leaving the compiled module intact.
Reloading executes the selected placement; it does not recreate an annotation
for a new optimization pass.

To deploy with ExecuTorch, save the same graph as a ``.pte``. This format does
not retrace by default:

.. code-block:: python

    torch_tensorrt.save(compiled, "model.pte", output_format="executorch")

Each TensorRT engine becomes a TensorRT delegate. An ``execute_in_torch`` region
runs on ExecuTorch's CPU kernels, like any operator TensorRT does not take,
unless a ``CudaPartitioner`` claims it (see :ref:`executorch_deployment`). A
``fused_op`` region calls the kernel the application registers for the
operator's ``out`` overload. TorchScript, AOTInductor, ``retrace=True``, and
``use_legacy_exporter=False`` are not supported for regions.

Built plugin engines include the compiled kernel, so loading them does not
require registering the Python kernel or plugin again. Plugin dry-run graphs
cannot be saved: they contain no built engine, and the exporter rejects them.

Models containing ``backend="inductor"`` regions cannot currently be saved or
exported through this path. The exporter serializes the reference FX operations,
which cannot preserve the installed Inductor callable. Such saves raise an
explicit error. Compile the original model again with ``backend="eager"`` if
you need the supported save/load workflow; loading a saved model does not enable
Inductor automatically.

The result of ``torch.compile`` cannot be saved this way. Compile with
``ir="dynamo"`` to get a model you can save.

.. _region_limits:

Supported scope and limitations
-----------------------------------

The initial API supports eval-mode functional inference with static shapes,
Tensor boundary values, read-only state, and one device per region. As in
PyTorch, 0-dim CPU tensors such as ``torch.tensor(2.0)`` or a scalar buffer can
be used with that device's tensors, except at ``fused_kernel`` boundaries,
which require CUDA tensors. :ref:`region_body` lists what the body of a region
may contain.

* Models containing live regions and complex tensors are rejected, including
  complex computations outside the region. Complex lowering does not yet
  preserve the region boundary correctly under either complex lowering mode.
* Nested scopes and graph breaks inside a scope are rejected.
* A region cannot contain, or be inside, a construct that export captures as
  a higher-order operator: ``torch.cond`` and other control-flow operators,
  ``flex_attention``, ``torch.autocast``, and grad-mode changes such as
  ``torch.no_grad()``. Use explicit casts such as ``.float()`` in place of
  ``torch.autocast``. A grad-mode block is captured only when it changes the
  grad mode in effect during compilation, so compile under ``torch.no_grad()``
  when a region contains or is inside a ``torch.no_grad()`` block.
* Compilation still traces region bodies through Torch-TensorRT's
  decompositions, so an operator whose decomposition raises an error fails
  compilation even inside a region.
* Disable CPU weight offload, ``enable_autocast``, resource partitioning,
  refit, and weight streaming. Conflicting dry runs return a reference graph
  and report conflicts without building engines or offloading state.
  ``require_full_compilation=True`` is incompatible with ``execute_in_torch``
  and ``fused_op`` regions because they must remain outside TensorRT.
  ``hardware_compatible=True`` is incompatible with ``fused_kernel`` regions
  because their kernel targets the compiling GPU's architecture.
* Empty or dead pure scopes are no-ops. A scope whose outputs are used only for
  their shape or dtype, as by ``torch.full_like``, can become dead during
  compilation, but it still counts as a live region for the settings above,
  static shapes, and complex tensors. A dead ``fused_op`` scope that draws
  random numbers is rejected.

These APIs select supplied implementations; they do not generate kernels.

Not supported yet
^^^^^^^^^^^^^^^^^

Each of these raises a ``RegionError`` or ``RegionCaptureError``, except where
noted. Unless noted, they apply to regions from scopes and from rules alike.
A live region is a region whose results the model uses.

* **Dynamic shapes.** A model with a live region needs static shapes on every
  path. A ``torch_tensorrt.Input`` with a shape range, a program exported with
  ``dynamic_shapes``, and ``torch.compile`` without ``dynamic=False`` on a new
  input shape all raise an error.
* **Refit.** ``MutableTorchTensorRTModule`` and ``immutable_weights=False``
  are rejected. Regions require ``immutable_weights=True``.
  ``refit_module_weights`` on a model compiled with regions raises an error
  that asks you to compile the model again with the new weights.
* **Engine caching.** ``cache_built_engines`` and ``reuse_cached_engines`` have
  no effect, without an error: caching needs ``immutable_weights=False``. A
  warning says that the engine will not be cached.
* **Writes to buffers or inputs.** A model with a live region must not write
  to a registered buffer or to an input, anywhere in the model. A key-value
  cache kept in a buffer is such a write. So are the running statistics of a
  batch norm in training mode. Every compile path raises an error that names
  the buffer or input; ``torch.compile`` names the graph input, such as
  ``l_x_``. Writes to a local temporary tensor outside a region remain
  supported.
* **Kernels across processes.** A ``fused_kernel`` scope must be compiled in
  the process that traced it. Use a ``FusedKernel`` rule in another process.
* **Training.** ``torch_tensorrt.compile(model, ...)`` rejects a model that
  has a module in training mode and names the modules. It trusts an eval-mode
  model: with ``ir="dynamo"``, dropout that the model keeps active on purpose
  outside its regions, as in ``F.dropout(x, training=True)``, compiles. A
  program you exported yourself does not say which mode its model was in. It
  is rejected when an operation outside its regions samples at random as a
  training-mode model does: dropout and its variants, attention with
  ``dropout_p`` above 0, ``rrelu`` with ``training=True``, and dropout between
  the layers of an RNN. ``torch.compile`` checks the mode of the model it
  compiles in the same way. When it compiles a function that calls the model,
  it applies the check for exported programs.
* **Engine-only output and cross-compilation.**
  ``torch_tensorrt.convert_method_to_trt_engine``,
  ``torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine``,
  and the Windows cross-compilation calls reject scopes and rules.
* **Decomposed programs.** A program with scopes that went through
  ``ExportedProgram.run_decompositions()`` is rejected.
* **Distributed tracing and higher-order transforms.** ``torch.compile`` with
  ``use_distributed_mode_trace=True``, and a ``torch.compile`` graph that uses
  ``torch.vmap``, are rejected.
* **The recompile limit of torch.compile.** After the limit, ``torch.compile``
  runs the model in PyTorch without its regions, and no error is raised. See
  :ref:`region_torch_compile`.
* **Other frontends.** ``regions=`` with an ``ir`` other than ``"dynamo"`` or
  ``"torch_compile"`` raises a ``ValueError``.

Inspect placement and measure performance
---------------------------------------------

To check placement, inspect ``compiled.graph`` and its submodules: each
``torch_tensorrt.runtime.TorchTensorRTModule`` is one TensorRT engine, and other
calls run in PyTorch. An engine's ``get_layer_info()`` lists its layers,
including ``fused_kernel`` plugins. For compiler development, the internal
manifest returned by
``torch_tensorrt.dynamo.region_compilation.get_region_records(compiled)``
records each region's placement; it can change without notice.

The result of ``torch.compile`` has no graph to inspect. On every path, the
debug log of the ``torch_tensorrt`` logger lists the module calls that rules
selected, in a line that starts with ``Region rules selected``.

For ``execute_in_torch``, capture extracts each region into a child FX graph
and makes it an opaque PyTorch call for partitioning. The child keeps the
operations export captured; Torch-TensorRT's TensorRT-oriented decompositions
do not rewrite it. Audits check that graph rewrites preserve the child and
keep it outside TensorRT partitions. ``fused_op`` uses the same boundary but
replaces the child body with a custom-op call. No capture marker executes at
runtime.

Each PyTorch region limits TensorRT fusion and can introduce extra
engine calls around a small block. Plugin regions keep the supplied kernel
inside TensorRT, but do not guarantee a speedup. Measure full-model latency and
engine counts against the same model without annotations. Fewer or larger
scopes may be faster; the annotation itself does not promise a performance
improvement.

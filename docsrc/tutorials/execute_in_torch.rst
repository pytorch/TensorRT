.. _execute_in_torch:

Region annotations: PyTorch, custom operators, and kernels
==========================================================

A region is a group of operations that you place yourself. Torch-TensorRT
compiles the rest of the model as usual. This experimental API has three kinds
of region:

* ``execute_in_torch`` keeps the operations together in PyTorch, even if they
  have TensorRT converters. ``backend="eager"``, the default, runs them as
  PyTorch operations. ``backend="inductor"`` compiles them with PyTorch
  Inductor.
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

Regions use private PyTorch APIs. If the installed PyTorch lacks them, every
trace that records a region raises an error whose message starts with
``Regions (execute_in_torch, fused_op, fused_kernel and regions= rules) require``.

Four runnable examples cover the API: :ref:`execute_in_torch_example`,
:ref:`fused_op_example`, :ref:`fused_kernel_example` and
:ref:`region_rules_example`. The plugin example requires Triton >=3.5 and
TensorRT QDP support; see
:ref:`the kernel API prerequisites <torch_tensorrt_kernels_py>`.

Mark a region and compile
-----------------------------

This model computes ``relu(x + 1) * 2``. Only the ReLU is marked to run in
PyTorch. Addition and multiplication remain eligible for TensorRT.

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

    compiled = torch_tensorrt.compile(model, inputs=[x], ir="dynamo", min_block_size=1)

    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x))

The execution order is::

    TensorRT addition -> PyTorch ReLU -> TensorRT multiplication

The optional ``name`` labels diagnostics. During ordinary eager execution the
scope does nothing: its body runs as ordinary code. A scope marks only the
operations captured inside it, so it can mark one call of a module that is
called several times; a rule selects every call.

.. _region_compile_paths:

Which compile paths honor regions
-------------------------------------

A scope written in ``forward`` is recorded whenever ``torch.export`` traces the
model, and whenever ``torch.compile`` traces it for the Torch-TensorRT backend.
The same compile calls accept ``regions=`` rules.

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
kernel given to ``fused_op`` or ``fused_kernel`` is not used. Pass rules to the
compile call, not to the trace: ``torch_tensorrt.dynamo.trace(model,
regions=...)`` raises a ``ValueError``.

Compile a region with Inductor
---------------------------------

Set ``backend="inductor"`` on the scope, or use
``ExecuteInTorch(backend="inductor")`` as a rule:

.. code-block:: python

    with torch_tensorrt.region.execute_in_torch(backend="inductor"):
        x = torch.relu(x)

The first call of the compiled model compiles the region with Inductor; later
calls reuse it, and a ``copy.deepcopy`` of the model compiles it again on the
copy's first call. Exclude the first call from timings. Compilation errors
reach the caller; nothing falls back to eager execution. Only ``"eager"`` and
``"inductor"`` are accepted: ``backend`` is the compiler choice, not the
``mode`` argument of ``torch.compile``.

Replace a region with a PyTorch custom operator
-----------------------------------------------

Use ``fused_op`` when an existing custom operator implements the region and
should execute through PyTorch. Register the operator and its fake
implementation before compiling.

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
                bias_relu_scale, inputs=(x, bias, scale), name="bias_relu_scale"
            ):
                x = torch.relu(x + bias) * scale
            return x - 0.5

The body is the reference calculation and runs during ordinary eager
execution. Compilation replaces the whole body with the operator; it does not
generate an implementation. The execution order is::

    TensorRT addition -> PyTorch custom operator -> TensorRT subtraction

* The operator must be a custom operator with positional Tensor arguments and
  Tensor results, and no mutation or aliasing. Its fake implementation must
  match the shapes and dtypes the body produces.
* ``inputs`` lists every tensor the region reads, including parameters and
  buffers, in the operator's argument order. Compilation cannot infer the
  order from the body. Omit ``inputs`` only when the region reads at most one
  tensor. Outputs follow the order the body produces them.
* The reference body may draw random numbers, for operators that own their
  RNG behavior. ``execute_in_torch`` and ``fused_kernel`` reject random
  operations.
* A traced program records the operator by name. The process that compiles
  the program must have registered the operator, or compilation raises a
  ``RegionError`` that names it.

Replace a region with a custom-kernel plugin
----------------------------------------------

Use ``fused_kernel`` when you have a kernel that implements the whole region.
For the Triton kernel and launch callbacks defined in
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

Torch-TensorRT uses the body to determine the kernel's inputs and outputs and
registers the operator and TensorRT plugin itself. The region must become part
of a TensorRT engine; compilation raises an error if it cannot.

* ``kernel`` is a zero-argument function that returns a ``@triton.jit`` kernel
  or a ``torch_tensorrt.kernels.KernelSpec`` for CUDA C++. ``kernel``,
  ``grid`` and ``extra_args_fn`` must be functions or ``functools.partial``
  objects, not methods. Strict export and ``torch.compile`` call them outside
  the trace, so a lambda written in ``forward`` must refer to globals; create
  other factories in ``__init__``.
* For Triton, the kernel's runtime parameters are the input pointers, optional
  ``i32`` extras, then the output pointers. Inputs follow ``inputs``; outputs
  are the tensors read after the scope, in the order the body produces them.
  A CUDA ``KernelSpec`` provides its own launch configuration.
* Inputs and outputs are static, contiguous CUDA tensors.
* The kernel is a Python object and is not saved with an exported program.
  Compile a program with a ``fused_kernel`` scope in the process that exported
  it, or select the module with a ``FusedKernel`` rule in another process.
* Each compilation registers a new operator and plugin for every
  ``fused_kernel`` region, for the lifetime of the process.
* You are responsible for the kernel computing what the body computes. Compare
  the compiled model with the original on representative inputs.

.. _region_body:

Write the body of a region
------------------------------

The body of a region is the code inside a scope, or the ``forward`` of a
module that a rule selects. It must be tensor code that the trace can record.

* **Inputs.** A region can read parameters, buffers and noncontiguous tensors,
  and inputs may share storage, such as the results of ``chunk`` or the
  weights of an ``nn.LSTM``. ``fused_op`` and ``fused_kernel`` regions receive
  a copy of an input that shares storage with an earlier one.
* **Casts.** Dtype casts such as ``.float()`` are supported inside a region and
  on its outputs, for example to compute a normalization in fp32 within an
  fp16 model.
* **Results.** A region can return several tensors. They must not share
  storage with the region's inputs or with each other. In a scope, clone a
  view before it leaves the scope. For a module that a rule selects,
  compilation makes the copy.
* **In-place operations.** An in-place operation is accepted when it writes a
  tensor that the region itself created and that nothing else reads, such as
  ``nn.ReLU(inplace=True)`` after a linear layer. A write to an input of the
  region raises ``Mutation or effects inside regions are unsupported``.
* **Operations that return their input.** Eval-mode dropout and a cast to the
  dtype a tensor already has are allowed in an ``execute_in_torch`` region. In
  a ``fused_op`` or ``fused_kernel`` scope whose result is read afterwards,
  clone the input inside the scope first. A rule treats a tensor that the
  module hands through unchanged as the caller's tensor.
* **Random operations.** Operations that draw random numbers are supported
  only in ``fused_op`` regions. Eval-mode dropout draws none.

.. _region_rules:

Select submodules with rules
--------------------------------

A rule places every call of a submodule in a region when you compile. Use
rules for a model whose source you cannot change, such as a Hugging Face
Transformers model, and for a program that is already exported.

.. code-block:: python

    from torch_tensorrt.region import ExecuteInTorch, FusedOp


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


    class Block(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.norm = Norm(width)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(width, 2 * width),
                torch.nn.SiLU(),
                torch.nn.Linear(2 * width, width),
            )

        def forward(self, x):
            return x + self.mlp(self.norm(x))


    class Layers(torch.nn.Module):
        def __init__(self, width=16, depth=2):
            super().__init__()
            self.layers = torch.nn.ModuleList(Block(width) for _ in range(depth))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x


    layers = Layers().eval().cuda()
    rules = {
        "layers.*.mlp": ExecuteInTorch(),
        Norm: FusedOp(scale_norm, inputs=[0, "weight"]),
    }
    compiled = torch_tensorrt.compile(
        layers, inputs=[x], ir="dynamo", min_block_size=1, regions=rules
    )

Each call of a selected module becomes one region. Here there are four: both
``norm`` modules call the operator, and both ``mlp`` modules run in PyTorch.
The model must be in eval mode.

Selectors
^^^^^^^^^

* A **path** is a name that ``model.named_modules()`` prints, such as
  ``"layers.0.mlp"``. ``*`` stands for one name and ``**`` for any number of
  names, so ``"layers.*.mlp"`` and ``"**.mlp"`` both select the two ``mlp``
  modules. Each name is matched as a shell pattern.
* A **class** selects every module of that class and of its subclasses.
* A **module object**, such as ``layers.layers[0].mlp``, selects that module.
  Use it when you compile the ``torch.nn.Module``.

An exported program records each module by its path and by the name of its
class, and holds no module objects. Select by path, or by class when the
compiling process has imported the class.

A selected module includes its children. Regions cannot nest or overlap, so
two rules that select the same operations, and a rule over a module whose
``forward`` holds a scope, raise a ``RegionCaptureError``. So does a rule that
matches no module; its message lists some of the model's module paths. A
container such as a ``ModuleList`` is never called: select its children with
``"layers.*"``.

The three rules
^^^^^^^^^^^^^^^

Each rule is the compile-time form of one scope:

* ``ExecuteInTorch(backend="eager", name=None)`` keeps the module in PyTorch.
* ``FusedOp(op, inputs=None, name=None)`` replaces the module with a custom
  operator.
* ``FusedKernel(kernel, inputs=None, name=None, ...)`` replaces the module with
  a kernel inside TensorRT. It takes the options of ``fused_kernel``.

``name`` defaults to the module's path.

Inputs and results of an operator or kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A rule has no tensors to list, so ``inputs`` lists positions and names, in the
order the operator or kernel takes its arguments:

* An integer is a position among the tensors that the module call receives:
  positional arguments first, then keyword arguments in the order the call
  passes them.
* A string names a parameter or buffer of the module, such as ``"weight"``, or
  ``"proj.weight"`` for a parameter of its child ``proj``.

``inputs=[0, "weight"]`` above passes the tensor that ``Norm`` receives, then
its ``weight``. List every tensor the module reads; omit ``inputs`` only when
the module reads one tensor. An incomplete list raises an error that names
what to add.

The operator or kernel returns what the module returns, in the order the
module returns it.

Positions and the return order need a record of how the module is called:

* ``torch_tensorrt.compile(model, ir="dynamo", regions=...)`` records it when
  it traces the module.
* A program you export yourself records it for the modules you name in
  ``torch.export.export(..., preserve_module_call_signature=("mix",))``.
* Otherwise, as with ``torch.compile``, the module must receive one tensor
  from its caller and the model must read one of its results. For other
  modules compilation raises a ``RegionCaptureError`` instead of guessing.

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

Compilation works on a copy of the program's graph. Your ``program`` stays as
it was exported, and you can compile it again with other rules.

.. _region_torch_compile:

Use regions with torch.compile
----------------------------------

The Torch-TensorRT ``torch.compile`` backend honors scopes written in
``forward``, and takes rules as a backend option. Pass ``dynamic=False``:

.. code-block:: python

    optimized = torch.compile(
        layers,
        backend="torch_tensorrt",
        dynamic=False,
        options={"min_block_size": 1, "regions": rules},
    )

``torch_tensorrt.compile(layers, ir="torch_compile", regions=rules)`` makes the
same call, passes ``dynamic=False`` for you, and checks the rules right away.

* **Errors appear on the first call**, where ``torch.compile`` traces and
  compiles. A region error from the backend arrives as
  ``torch._dynamo.exc.BackendCompilerFailed``, whose ``inner_exception`` is the
  ``RegionError`` or ``RegionCaptureError``.
* **Nothing falls back silently.** If a graph with regions cannot be compiled,
  the call raises a ``RegionError`` instead of running the model in PyTorch
  without its regions.
* **Shapes are static.** With ``dynamic=False`` each new input shape is
  compiled on its own. Without it, a call with a new shape raises an error
  that asks for ``dynamic=False``.
* **The recompile limit.** ``torch.compile`` compiles a model for a limited
  number of input shapes, ``torch._dynamo.config.recompile_limit``, 8 by
  default. After that it logs ``hit config.recompile_limit`` and runs further
  shapes in PyTorch, without the regions and without an error. Set
  ``torch._dynamo.config.fail_on_recompile_limit_hit = True`` to get an error
  instead.
* **Graph breaks.** A graph break between scopes is fine. A graph break inside
  a scope is an error. A rule needs the whole call of its module in one
  graph.
* **Paths start at the model you compiled.** If you compile a function, such as
  ``torch.compile(model.forward, ...)``, select by class or by module object.
* **Blocks.** A region inside a ``torch.autocast`` block, or inside a
  ``torch.no_grad()`` block that changes the grad mode, is rejected. Enter the
  block around the call of the compiled model, or use explicit casts.

.. _region_export_later:

Export first, compile later
-------------------------------

``torch.export.export`` records the scopes written in ``forward``, with
``strict=True`` or without. Compile the program whenever you want:

.. code-block:: python

    program = torch.export.export(model, (x,))
    compiled = torch_tensorrt.dynamo.compile(program, inputs=[x], min_block_size=1)

The program holds two marker operators around each scope,
``torch_tensorrt_region::begin`` and ``torch_tensorrt_region::end``. They do
nothing when the program runs in PyTorch.

To compile in another process, save the program with ``torch.export.save``.
In the other process, import ``torch_tensorrt`` before ``torch.export.load``:
the import registers the marker operators, and loading fails without it. A
``fused_op`` scope also needs its operator registered there.

Do not call ``program.run_decompositions()`` before you compile. It rewrites
the markers, and compilation then raises a ``RegionCaptureError``.

.. _region_ignore:

Export for another tool
^^^^^^^^^^^^^^^^^^^^^^^

Another tool that reads exported programs may not know the markers. For
example, ``torch.onnx.export(model, (x,), dynamo=True)`` fails on them. Trace
the model inside ``ignore_regions`` to record its scopes as ordinary
operations:

.. code-block:: python

    with torch_tensorrt.region.ignore_regions():
        plain_program = torch.export.export(model, (x,))

``torch.compile`` traces on the first call of the compiled model, so the block
must surround that call. The block does not affect ``regions=`` rules.

.. _region_limits:

Limitations
---------------

Regions support eval-mode inference with static shapes, read-only state and
one device per region. Each of the following raises a ``RegionError`` or
``RegionCaptureError``, unless noted.

* **Dynamic shapes.** A ``torch_tensorrt.Input`` with a shape range, a program
  exported with ``dynamic_shapes``, and ``torch.compile`` on a new input shape
  without ``dynamic=False``.
* **Writes to buffers or inputs**, anywhere in a model with a live region. A
  key-value cache kept in a buffer is such a write. Writes to a local
  temporary tensor outside a region remain supported.
* **Training mode.** A model with a module in training mode, and an exported
  program that samples at random outside its regions as a training-mode model
  does.
* **Settings.** ``offload_module_to_cpu``, ``enable_autocast``, resource
  partitioning, weight streaming, ``strip_engine_weights`` and
  ``immutable_weights=False``. ``require_full_compilation=True`` conflicts
  with ``execute_in_torch`` and ``fused_op`` regions, and
  ``hardware_compatible=True`` with ``fused_kernel`` regions. A dry run raises
  the same errors.
* **Refit.** ``MutableTorchTensorRTModule`` and ``refit_module_weights``.
  Compile the model again with the new weights.
* **Saving.** ``torch_tensorrt.save`` and ``torch_tensorrt.executorch.export``
  of a model compiled with regions. Save the exported program instead and
  compile it where it runs.
* **CUDA graphs.** ``torch_tensorrt.runtime.enable_cudagraphs`` on a model with
  ``execute_in_torch`` or ``fused_op`` regions.
* **Nesting.** Nested scopes, and a region that contains or sits inside a
  construct that export captures as a higher-order operator, such as
  ``torch.cond`` or ``torch.autocast``.
* **Complex tensors** anywhere in a model with a live region.
* **Engine-only output and cross-compilation.**
  ``torch_tensorrt.convert_method_to_trt_engine`` and the Windows
  cross-compilation calls reject a model with scopes. They take no
  ``regions=`` option, so passing one raises a ``TypeError``.
* **Decomposed programs.** A program with scopes that went through
  ``ExportedProgram.run_decompositions()``.
* **Other frontends.** ``regions=`` with an ``ir`` other than ``"dynamo"`` or
  ``"torch_compile"`` raises a ``ValueError``.

Without an error: after the recompile limit of ``torch.compile``, further
input shapes run in PyTorch without their regions (see
:ref:`region_torch_compile`), and engine caching has no effect because it
needs ``immutable_weights=False``.

Inspect placement
---------------------

Inspect ``compiled.graph`` and its submodules: each
``torch_tensorrt.runtime.TorchTensorRTModule`` is one TensorRT engine, and
other calls run in PyTorch. An engine's ``get_layer_info()`` lists its layers,
including ``fused_kernel`` plugins. The result of ``torch.compile`` has no
graph to inspect; the debug log of the ``torch_tensorrt`` logger lists the
module calls that rules selected, in a line that starts with
``Region rules selected``.

Each PyTorch region limits TensorRT fusion and can add engine calls around a
small block. Measure full-model latency against the same model without
regions.

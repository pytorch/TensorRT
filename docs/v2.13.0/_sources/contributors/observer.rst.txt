.. _observer:

Observer / Callback System
===========================

Torch-TensorRT exposes a lightweight instrumentation hook — the ``Observer`` / ``@observable``
system — that lets you attach callbacks to internal compiler functions **without modifying
source code**. This is useful for:

* Logging intermediate graph states during development.
* Collecting timing or shape statistics across compilation stages.
* Writing integration tests that assert on internal state.
* Capturing conversion context snapshots for debugging.

The system lives in ``py/torch_tensorrt/dynamo/observer.py`` and is thread- and
context-safe via Python ``contextvars``.

----

Core Concepts
--------------

``Observer``
    A named, context-variable-backed event source. Any function can hold an
    ``Observer`` instance and call ``observer.observe(*args)`` at a point of interest.
    Registered callbacks receive those arguments.

``@observable()``
    A decorator that wraps a function and automatically fires ``pre`` and ``post``
    observers around each invocation, passing an ``ObserveContext`` to each callback.

``ObserveContext``
    Dataclass passed to every ``@observable`` callback with fields:

    * ``callable`` — the observed function object.
    * ``args`` — positional arguments passed to it.
    * ``kwargs`` — keyword arguments passed to it.
    * ``return_value`` — return value (only populated in ``post`` callbacks).

----

Using ``Observer`` Directly
-----------------------------

Any internal function can expose an ``Observer`` instance as a module-level constant:

.. code-block:: python

    # inside a Torch-TensorRT module (illustrative)
    from torch_tensorrt.dynamo.observer import Observer

    on_subgraph_converted: Observer = Observer(name="on_subgraph_converted")

    def convert_subgraph(gm, settings):
        result = _do_convert(gm, settings)
        on_subgraph_converted.observe(gm, result)   # fire the event
        return result

Consumers attach a callback using the context-manager form of ``Observer.add()``:

.. code-block:: python

    from torch_tensorrt.dynamo.observer import Observer

    records = []

    def capture(gm, result):
        records.append((gm.graph, result.serialized_engine))

    with on_subgraph_converted.add(capture):
        trt_gm = torch_tensorrt.dynamo.compile(exported_program, arg_inputs=inputs)

    print(f"Captured {len(records)} subgraph conversions")

The callback is automatically de-registered when the ``with`` block exits, so it is
only active for the compilation inside the block. Callbacks registered outside a
``with`` statement are not automatically removed — call ``observer._get_callbacks().remove(callback)``
manually.

----

Using ``@observable``
-----------------------

``@observable()`` is more ergonomic when you own the function being observed. It adds
a ``.observers`` attribute of type ``CallableObservers`` with two sub-observers:
``.pre`` (fires before the function) and ``.post`` (fires after, with the return value):

.. code-block:: python

    from torch_tensorrt.dynamo.observer import observable, ObserveContext

    @observable()
    def my_lowering_pass(gm, settings):
        # ... apply transformations ...
        return gm

    # Attach a pre-callback
    def log_before(ctx: ObserveContext):
        print(f"About to run {ctx.callable.__name__} on graph with "
              f"{len(list(ctx.args[0].graph.nodes))} nodes")

    # Attach a post-callback
    def log_after(ctx: ObserveContext):
        print(f"After pass: graph has "
              f"{len(list(ctx.return_value.graph.nodes))} nodes")

    with my_lowering_pass.observers.pre.add(log_before), \
         my_lowering_pass.observers.post.add(log_after):
        my_lowering_pass(gm, settings)

----

Error Handling
--------------

By default, exceptions raised inside a callback are **caught and logged** at ``INFO``
level; they do not propagate to the caller. This ensures that instrumentation bugs
never crash a production compilation.

During unit tests you can opt in to exception re-raising:

.. code-block:: python

    import torch_tensorrt.dynamo.observer as obs_module

    obs_module.RETHROW_CALLBACK_EXCEPTION = True
    # ... run test ...
    obs_module.RETHROW_CALLBACK_EXCEPTION = False  # restore

----

Concurrency and Context Isolation
-----------------------------------

Callbacks are stored in a ``contextvars.ContextVar``, which means each Python
execution context (thread, async task, ``concurrent.futures`` worker) has its own
callback registry. A callback registered in one thread is not visible to observers
firing in another thread. This makes the system safe to use in multi-threaded
compilation pipelines.

----

Observing Existing Compiler Functions
----------------------------------------

To observe a function that you do not own (i.e., you cannot decorate it), wrap it with
``_make_observable`` and monkey-patch:

.. code-block:: python

    from torch_tensorrt.dynamo import observer as _obs
    import torch_tensorrt.dynamo._compiler as _compiler

    # Wrap the target function
    original = _compiler.compile_module
    _compiler.compile_module = _obs._make_observable(original)

    records = []
    with _compiler.compile_module.observers.post.add(
        lambda ctx: records.append(ctx.return_value)
    ):
        result = torch_tensorrt.dynamo.compile(exported_program, arg_inputs=inputs)

    # Restore
    _compiler.compile_module = original

.. warning::

   Monkey-patching internal functions is fragile across Torch-TensorRT versions.
   Prefer using the official ``Observer`` instances exposed by each module when
   available.

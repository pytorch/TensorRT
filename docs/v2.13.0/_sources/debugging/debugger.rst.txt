.. _debugger:

Debugger
========

The ``torch_tensorrt.dynamo.Debugger`` context manager provides visibility into
the Torch-TensorRT compilation pipeline. Use it to capture FX graph
visualizations around lowering passes, monitor engine building, save profiling
data, and control logging verbosity — all without modifying your model code.

Usage
-----

Wrap your ``torch_tensorrt.dynamo.compile`` (or ``torch.compile``) call inside
the ``Debugger`` context:

.. code-block:: python

    import torch_tensorrt

    with torch_tensorrt.dynamo.Debugger(
        log_level="debug",
        logging_dir="/tmp/trt_debug",
        engine_builder_monitor=True,
        capture_fx_graph_before=["remove_detach"],
        capture_fx_graph_after=["complex_graph_detection"],
    ):
        trt_model = torch_tensorrt.dynamo.compile(exported_program, arg_inputs=inputs)
        output = trt_model(*inputs)

On exit the context restores all previous logging state and lowering-pass
lists automatically.

Options
-------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``log_level``
     - ``"debug"``
     - Verbosity level: ``"debug"``, ``"info"``, ``"warning"``, ``"error"``,
       ``"internal_errors"``, or ``"graphs"``.
   * - ``logging_dir``
     - ``/tmp/torch_tensorrt_<user>/debug_logs``
     - Directory where logs, profiles, and graph SVGs are written.
   * - ``capture_fx_graph_before``
     - ``None``
     - List of lowering-pass names. An SVG of the FX graph is saved
       *before* each named pass runs.
   * - ``capture_fx_graph_after``
     - ``None``
     - List of lowering-pass names. An SVG of the FX graph is saved
       *after* each named pass runs.
   * - ``engine_builder_monitor``
     - ``True``
     - Stream TensorRT engine-build progress to the console.
   * - ``save_engine_profile``
     - ``False``
     - Save per-layer profiling data after the first inference run.
   * - ``profile_format``
     - ``"perfetto"``
     - Format for saved profiles. C++ runtime supports ``"perfetto"`` and
       ``"trex"``; Python runtime supports ``"cudagraph"`` only.
   * - ``capture_tensorrt_api_recording``
     - ``False``
     - Record TensorRT API calls for replay-based debugging (Linux only).
       Requires ``TORCHTRT_ENABLE_TENSORRT_API_CAPTURE=1``.
   * - ``save_layer_info``
     - ``False``
     - Write a JSON file containing per-layer metadata for the compiled
       engine.

Output layout
-------------

After the context exits, ``logging_dir`` contains:

.. code-block:: text

    <logging_dir>/
        torch_tensorrt_logging.log          # always written
        lowering_passes_visualization/      # when capture_fx_graph_before/after used
            before_<pass_name>.svg
            after_<pass_name>.svg
        engine_visualization_profile/       # when save_engine_profile=True
        engine_layer_info.json              # when save_layer_info=True

TensorRT API recordings (``capture_tensorrt_api_recording=True``) are written
separately to ``/tmp/torch_tensorrt_<user>/shim/`` and are independent of
``logging_dir``.

API Reference
-------------

.. currentmodule:: torch_tensorrt.dynamo

.. autoclass:: Debugger
   :members:
   :undoc-members:
   :show-inheritance:

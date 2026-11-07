Optional local MCP adapter
==========================

The separately installable ``torch-tensorrt-mcp`` package exposes nine tools over
stdio. Heavy GPU imports and model execution occur inside spawned workers. The
server provides no HTTP listener, arbitrary path reader, deletion operation,
shell execution, or dependency/environment mutation tool.

The v1 execution supervisor supports Linux. Its restart recovery uses Linux
process identity checks before terminating orphaned workers; other operating
systems are rejected before any job state is created.

Installation and compatibility
------------------------------

Provision Torch, TensorRT, CUDA, and a compatible Torch-TensorRT build first,
using the project's normal installation instructions. Then run this command
from the repository root in that environment:

.. code-block:: console

   python -m pip install ./tools/mcp

The default adapter requirements include MCP v2, Pydantic, and Packaging. They
do not include Torch, TensorRT, or Torch-TensorRT. The explicit ``[core]`` extra
is for provisioning a new environment and can change its GPU dependencies; it
is not needed to connect to an existing installation. Launch never installs or
upgrades packages.

.. list-table:: Adapter compatibility
   :header-rows: 1
   :widths: 20 30 50

   * - Component
     - Admitted version
     - Verification requirement
   * - Adapter
     - ``0.1.0``
     - Experimental report schema (``0.1``)
   * - Torch-TensorRT or Torch-TensorRT RTX
     - ``>=2.16.0.dev0``
     - Build must contain public reporting, artifacts, analysis, compile-report,
       and verification APIs; the numeric range alone is insufficient.
   * - Official MCP Python SDK
     - ``>=2,<3``
     - Stable ``2.3.0`` is the reference test version; CPU CI also exercises the
       latest admitted v2 release.
   * - Python
     - ``>=3.11,<3.15``
     - CPU CI runs 3.11 and 3.12; GPU compatibility follows the provisioned core.

This table distinguishes admission checks from a claim that every build in a
range has been GPU-tested. Execution fails with retained diagnostics when core
metadata or required APIs are unavailable. TensorRT engine compatibility still
depends on the recorded environment and compilation settings.

Configure trusted factories
---------------------------

Create a trusted model directory and a separate output directory before launch.
For example, save the following as ``/srv/torchtrt-models/factories.py``:

.. code-block:: python

   import torch
   from torch_tensorrt.verification import InputSample

   class Add(torch.nn.Module):
       def forward(self, x):
           return x + x

   def model():
       return Add().cuda().eval()

   def inputs(seed, batch_size=1):
       generator = torch.Generator(device="cuda").manual_seed(seed)
       x = torch.randn(
           batch_size, 3, device="cuda", dtype=torch.float16,
           generator=generator,
       )
       return InputSample(args=(x,), profile="static")

Save an administrator configuration such as
``/srv/torchtrt-models/admin-config.json``:

.. code-block:: json

   {
     "workspace_root": "/srv/torchtrt-models",
     "output_root": "/srv/torchtrt-jobs",
     "models": {
       "add": {"path": "factories.py", "function": "model", "parameters": []}
     },
     "inputs": {
       "add": {"path": "factories.py", "function": "inputs", "parameters": ["batch_size"]}
     },
     "max_pending_jobs": 32,
     "max_timeout_seconds": 3600,
     "max_response_bytes": 262144
   }

The model factory returns ``nn.Module`` or an administrator-exported
``ExportedProgram``. Returning an exported program supplies dynamic capture
constraints; export it in evaluation mode. A module is evaluated and exported
using the first input sample. The input factory receives ``seed`` and explicitly
allowlisted JSON keyword parameters. It returns one ``InputSample``, a nonempty
list of them, an argument tuple, or a dictionary with ``args``, ``kwargs``,
``profile``, ``sample_id``, and ``label`` fields. Actual tensors determine the
device and strongly typed execution precision. The adapter does not cast tensors
or require a graph-wide dtype; converter support determines which explicit dtype
combinations can be compiled.

Client setup
------------

Use the full executable path from the provisioned environment so the client
does not select a different Python installation. Launch directly with:

.. code-block:: console

   /srv/torchtrt-env/bin/torch-tensorrt-mcp --config /srv/torchtrt-models/admin-config.json

For Cursor, put this stdio entry in the project's ``.cursor/mcp.json`` or your
personal ``~/.cursor/mcp.json``. These locations and fields follow the
`Cursor MCP documentation <https://prod.cursor.com/docs/mcp>`_.

.. code-block:: json

   {
     "mcpServers": {
       "torch-tensorrt": {
         "type": "stdio",
         "command": "/srv/torchtrt-env/bin/torch-tensorrt-mcp",
         "args": ["--config", "/srv/torchtrt-models/admin-config.json"]
       }
     }
   }

For Claude Code, register the same executable with the documented
`stdio command syntax <https://code.claude.com/docs/en/mcp#option-3-add-a-local-stdio-server>`_:

.. code-block:: console

   claude mcp add --transport stdio torch-tensorrt -- /srv/torchtrt-env/bin/torch-tensorrt-mcp --config /srv/torchtrt-models/admin-config.json

The configuration is local to this client; it does not grant the adapter access
to arbitrary model names or expand the administrator's allowed settings.

Compile, reload, and inspect evidence
-------------------------------------

Each tool receives one ``request`` object. ``collect_environment``, ``analyze``,
``compile``, ``validate``, and ``benchmark`` enqueue work and immediately return
a job ID. The remaining tools inspect state, request cancellation, or retrieve
completed evidence. ``collect_environment`` uses ``{"request": {}}``.

Submit ``analyze`` and then ``compile`` with:

.. code-block:: json

   {
     "request": {
       "model": "add",
       "inputs": "add",
       "input_parameters": {"batch_size": 1},
       "seed": 0,
       "settings": {"min_block_size": 1}
     }
   }

Poll ``get_job_status`` with ``{"request": {"job_id": "<job-id>"}}``. Adapter
job IDs are 32 lowercase hexadecimal characters. After the compilation job
succeeds, submit two new jobs referring to its immutable ID:

.. code-block:: json

   {
     "request": {
       "source_job_id": "<compile-job-id>",
       "atol": 0.001,
       "rtol": 0.001,
       "relative_floor": 0.000000000001
     }
   }

The object above is for ``validate``. For ``benchmark``:

.. code-block:: json

   {
     "request": {
       "source_job_id": "<compile-job-id>",
       "sample_index": 0,
       "warmup": 10,
       "iterations": 100,
       "percentiles": [50.0, 90.0, 99.0],
       "batch_size": 1
     }
   }

The supervisor verifies the original request and artifact against the compile
manifest. A fresh worker reloads ``artifacts/compiled.pt2``, rebuilds the original
configured reference and input samples, and checks factory source, graph,
constraints, weights, and input hashes. Follow-up requests cannot replace the
original model, generator, parameters, or seed. Validation tests all generated
samples; benchmark chooses one by ``sample_index`` (default zero). Its report
retains the selected index, profile, sample ID, label, and declared GPU isolation.
The MCP adapter cannot verify system-wide GPU exclusivity, so that protocol field
remains ``unknown``.
For dynamic profiles, configure an exported program with the desired constraints
and return named min/opt/max samples from the original input factory.

Read ``get_report`` with the job ID and a named report: ``environment``,
``analysis``, ``compilation``, ``validation``, ``benchmark``, or ``failure``.
``get_artifact_manifest`` takes the job ID. Both are available after worker exit;
reports are checked against their manifest file hash. Missing stage reports have
explicit reasons. Oversized responses return omission metadata, leaving the full
manifest-linked file available locally to its owner.

Policy and lifecycle
--------------------

Only these compilation settings are accepted:

.. list-table:: JSON compilation settings
   :header-rows: 1
   :widths: 45 55

   * - Setting
     - Accepted value
   * - ``min_block_size``
     - Integer 1--100000
   * - ``optimization_level``
     - ``null`` (compiler default) or integer 0--5
   * - ``workspace_size``
     - Integer 0--64 GiB, in bytes
   * - ``num_avg_timing_iters``
     - Integer 1--100
   * - ``max_aux_streams``
     - ``null`` or integer 0--1024
   * - ``require_full_compilation``, ``disable_tf32``, ``use_fp32_acc``,
       ``version_compatible``, ``hardware_compatible``, ``truncate_double``,
       ``use_fast_partitioner``, ``enable_experimental_decompositions``,
       ``assume_dynamic_shape_support``, ``sparse_weights``,
       ``offload_module_to_cpu``, ``decompose_attention``,
       ``fallback_data_dependent_ops``
     - Boolean

Unknown fields, nonfinite JSON, request-supplied code/callables, cache/save paths,
calibration dataloaders, and dependency mutation are rejected. The worker assigns
timing-cache and diagnostic paths inside its job directory, and disables shared
engine-cache reads and writes.

Allowlisted factories are trusted executable Python running with the server's
OS credentials. The adapter is a policy boundary for client requests, not an OS
sandbox for factory code or its imports. Protect configuration, trusted source,
and output directories from untrusted writers. Each execution worker loads the
current factory source, so edits take effect without restarting the server.
Factory source files are not hashed or pinned by the adapter. Transitive imports
and external model files retain normal Python permissions.

One spawned execution worker runs at a time per output root; a second adapter
cannot concurrently manage the same root. Timeouts begin when execution starts.
Cancellation transitions through ``cancellation_requested`` and becomes
``cancelled`` after child exit. Termination escalates after one second.
Supervisor-owned finalization preserves partial diagnostics even if the child
crashes or is killed. Restart fails unfinished jobs rather than resuming engine
builds. Linux orphan matching checks process start time; platforms without
``/proc`` cannot safely identify workers left by a crashed server.

Live records reside in ``.jobs/`` and job files in ``.<job_id>.tmp``. After the
worker exits, the supervisor writes a manifest and publishes the directory under
the human-readable name returned as ``output_dir`` by ``get_job_status``, for
example ``20261104_153000_compile_example_0123abcd/``. Follow-up manifests record
``source_job_id`` without changing the compilation directory. Administrators
manage retention outside this tool surface.

Troubleshooting
---------------

If tool discovery fails, check the absolute executable/config paths and client
stderr. Discovery does not import the GPU core. If an execution job fails, poll
its terminal status and inspect its categorized ``failure`` report and retained
``diagnostics/`` logs. A missing core or public API requires a compatible
provisioned build; restarting never fixes package dependencies automatically.

A provenance failure means the reference graph, constraints, weights, or inputs
can no longer be reconstructed exactly. Review the factory, weights, and
generator, then submit a new compilation. Reusing a source job does not bypass
these checks. A validation
failure supplies output paths and tested samples; adjust tolerances only after
checking the numerical evidence and the intended accuracy requirement.

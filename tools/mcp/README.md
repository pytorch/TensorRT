# Torch-TensorRT MCP adapter

This optional Linux package exposes nine local tools using the official MCP Python SDK v2 over stdio. The tested SDK version is 2.3.0. The core package has no MCP dependency. The adapter imports GPU libraries only inside spawned execution workers. The v1 job manager rejects non-Linux hosts because its audited orphan-worker recovery requires `/proc` process identities and Linux process groups.

Install this package into a **previously provisioned Torch/TensorRT/Torch-TensorRT environment**:

```sh
python -m pip install ./tools/mcp
```

For a complete Cursor Agent integration, see
[`examples/cursor/README.md`](examples/cursor/README.md).

The default package dependencies contain MCP, Pydantic, and Packaging. They do not contain Torch, TensorRT, or Torch-TensorRT, so the package resolver cannot replace those packages. An explicit `[core]` extra exists for new-environment provisioning. Jobs verify the installed core distribution is `>=2.16.0.dev0` and requires the new public reporting/verification APIs; installations in that range without those APIs fail with retained diagnostics. The launch command never installs dependencies or alters the environment.

Create `workspace_root` before launch. The adapter creates `output_root`,
including missing parent directories, when it loads the configuration. Example
administrator configuration:

```json
{
  "workspace_root": "/absolute/trusted/models",
  "output_root": "/absolute/job-output",
  "models": {
    "example": {"path": "factories.py", "function": "model", "parameters": []}
  },
  "inputs": {
    "example": {"path": "factories.py", "function": "inputs", "parameters": ["batch_size"]}
  },
  "max_pending_jobs": 32,
  "max_timeout_seconds": 3600,
  "max_response_bytes": 262144
}
```

The model factory takes its allowlisted JSON keyword arguments and returns an `nn.Module` or an already captured `torch.export.ExportedProgram`. Export in evaluation mode when returning an exported program. The input factory additionally receives `seed` and returns an argument tuple, `InputSample`, a nonempty list of `InputSample`, or a dictionary containing `args`, `kwargs`, `profile`, `sample_id`, and `label` fields. Actual input tensors must be prepared on the intended device. Dynamic export constraints come from an administrator-provided exported program; the input factory can supply min/opt/max profile samples for validation. Benchmark selects one sample with `sample_index` (default 0). Requests cannot inject code, imports, callables, export profiles, cache paths, save destinations, or unverified GPU-isolation claims.

Model and input factories must prepare the intended tensor dtypes. The adapter does not cast tensors or enforce a graph-wide precision: the strongly typed graph and its converters determine supported dtype combinations. Compilation uses the metadata exporter (`retrace=False`) to retain dynamic guards. Follow-up jobs verify that the configured factory identities remain authorized, then compare the reconstructed graph, range constraints, model weights, prepared inputs, and sample metadata against compilation provenance before executing the saved artifact.

The JSON settings allowlist also exposes bounded partitioning and builder controls: full compilation, TF32/FP32 accumulation, optimization level, workspace size, version/hardware compatibility, double truncation, fast or global partitioning, experimental decompositions, dynamic-shape support, sparse weights, timing iterations, auxiliary streams, CPU offload, attention decomposition, and data-dependent-op fallback. Paths, Python objects, custom operators, and shared cache controls remain adapter-owned or prohibited.

Factories are **trusted executable Python**, and allowlisting them is authorization to execute the current contents of those files with the server's OS credentials. This adapter is not an OS sandbox for malicious factories. Factory source is loaded afresh in each execution worker, so edits take effect without restarting the server; source files are not hashed or pinned by the adapter. Model/input factories from the same source share one imported module per worker, supporting dataclass and pytree identities. The trusted workspace and source-file directory are added to the worker's import path for sibling imports. Parameters must be explicitly listed by the administrator. Keep configuration, model code, transitive imports, and output roots writable only by trusted users. File checks reject traversal and symlinks, but factory code and its imports retain ordinary Python permissions.

Run:

```sh
torch-tensorrt-mcp --config /absolute/admin-config.json
```

For Claude Code or Cursor, add an MCP stdio server using the full path to the environment's `torch-tensorrt-mcp` executable and the arguments `--config`, `/absolute/admin-config.json`. For example:

```json
{
  "mcpServers": {
    "torch-tensorrt": {
      "command": "/absolute/venv/bin/torch-tensorrt-mcp",
      "args": ["--config", "/absolute/admin-config.json"]
    }
  }
}
```

Each execution tool takes one `request` object. Start by checking the provisioned environment:

```json
{"request": {}}
```

Then call `analyze` or `compile` with a configured model and inputs:

```json
{"request": {"model": "example", "inputs": "example", "input_parameters": {"batch_size": 1}, "seed": 0, "settings": {"min_block_size": 1}}}
```

Tools return a `job_id` immediately. Poll `get_job_status` with `{"request":{"job_id":"<32 lowercase hex characters>"}}`. After a compile job succeeds, start validation or benchmarking against its saved artifact:

```json
{"request": {"source_job_id": "<compile job_id>"}}
```

Both follow-up tools accept additional bounded options; for example, benchmark accepts `sample_index`, `warmup`, `iterations`, `percentiles`, and `batch_size`, while validation accepts `atol`, `rtol`, and `relative_floor`. Every execution request also accepts an optional `timeout_seconds`, subject to the administrator's configured maximum.

`get_report` accepts a job ID and one named report (`environment`, `analysis`, `compilation`, `validation`, `benchmark`, `failure`). `get_artifact_manifest` lists local artifact paths, hashes, and lineage. Large reports receive bounded metadata instead of truncated JSON. No arbitrary path-reading, deletion, shell, or dependency-mutation tool is exposed.

Concurrency is one worker per output root, including across server instances. Timeout starts when execution begins. Cancellation transitions through `cancellation_requested` and becomes terminal after the worker exits; termination escalates after one second. On restart, nonterminal jobs are failed and matching orphan process groups are stopped. The adapter never resumes a half-built TensorRT engine. Shutdown cancels remaining jobs. Linux orphan identification includes process start time to avoid PID reuse; additional operating systems require a future trustworthy process-identity implementation.

Live state is stored in `.jobs/`, with append-only events and worker diagnostics in `.<job_id>.tmp`. After child exit, the supervisor hashes retained files, writes the manifest, and publishes the directory under the human-readable name returned as `output_dir` by `get_job_status` (for example, `20261104_153000_compile_example_0123abcd/`). Completed directories are immutable through this API. Compile writes `artifacts/compiled.pt2`; follow-up jobs reload that exact manifest-verified artifact and reconstruct the original configured reference/inputs. Administrators manage retention outside the tool surface.

Test the adapter independently of CUDA:

```sh
python -m pytest -c tools/mcp/pyproject.toml tools/mcp/tests
```

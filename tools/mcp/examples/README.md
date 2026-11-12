# Torch-TensorRT MCP examples

The administrator configuration and model factories in this directory are
platform-neutral and can be used by any client that supports stdio MCP
servers. `run_with_cursor.py` is an optional Cursor SDK client: it starts a
Cursor Local Agent, gives it access to the configured Torch-TensorRT MCP
server, streams the Agent's response, and reports token usage for that SDK run.

The script can ask the MCP server to analyze, compile, validate, or benchmark
models allowlisted in `admin-config.json`. Each invocation gets an isolated
run directory under `.mcp-runs/` by default. Its name contains
the UTC start time and an eight-character random identifier, for example
`20261105_011204_e04a1d83`. This keeps runs sortable and unique without sharing
the MCP job-manager lock used by the Cursor IDE.

## Prerequisites

- Linux with a working Torch, TensorRT, and Torch-TensorRT environment.
- The Torch-TensorRT MCP adapter installed in that environment.
- Python 3.10 or newer.

Connect any MCP-capable agent platform to `torch-tensorrt-mcp` with
`--config /absolute/path/to/tools/mcp/examples/admin-config.json`. The
following requirements apply only when using the optional Cursor SDK client:

- A Cursor API key from [Cursor Dashboard → API Keys](https://cursor.com/dashboard/api-keys).
- A valid stdio MCP entry in `.cursor/mcp.json`.

Install the Cursor SDK from the repository root:

```bash
python -m pip install -r tools/mcp/examples/requirements.txt
```

The script is tested with `cursor-sdk 1.3.1`. Check the installed version with:

```bash
python -m pip show cursor-sdk
```

## Configure the MCP server

The default MCP configuration is `.cursor/mcp.json`. Its server name must be
`torch-tensorrt` unless a different name is passed with `--mcp-server`.

```json
{
  "mcpServers": {
    "torch-tensorrt": {
      "type": "stdio",
      "command": "/absolute/path/to/bin/torch-tensorrt-mcp",
      "args": [
        "--config",
        "/absolute/path/to/TensorRT/tools/mcp/examples/admin-config.json"
      ],
      "env": {
        "LD_LIBRARY_PATH": "/absolute/path/to/TensorRT/lib"
      }
    }
  }
}
```

The configured `LD_LIBRARY_PATH` must contain the TensorRT libraries required
by the worker, including `libnvinfer.so` and `libnvonnxparser.so`. The SDK
starts a new MCP process and passes this configured environment to it; do not
assume it will use the environment of an already-running Cursor IDE MCP server.

Models and input factories must be allowlisted in
`tools/mcp/examples/admin-config.json`. Update its repository paths
before launching the server. The configured `workspace_root` must exist; the
server creates `output_root` if needed. Generated `.mcp-runs/` artifacts are
ignored by Git.
The server creates `<output_root>/.graph_transformation_candidates` for
generated graph transformations.
The Agent selects one through the MCP request's reserved
`graph_transformation_candidate_id` field. Model factories do not accept or
need to know about that field.
Candidate files are trusted, unrestricted Python and execute with the worker's
OS credentials. They are not sandboxed; only trusted users and Agents should
receive both write access to the candidate directory and permission to select
a candidate ID.
The included examples define the model and input names `add`, `qwen3_5`,
`batched_linear`, and `diff`. The `diff` example intentionally retains the
currently unsupported `aten.diff.default` operator for custom decomposition
experiments.

## Set the Cursor API key

Avoid placing the key in shell history:

```bash
read -rsp "Cursor API Key: " CURSOR_API_KEY
export CURSOR_API_KEY
echo
```

Do not commit the key, put it in MCP JSON, or paste it into logs or chat.

## Choose a Cursor model

The script defaults to `--model auto`. Some teams disable the corresponding
Cursor Router "Optimize For" mode. List the models available to the current
API key:

```bash
python - <<'PY'
from cursor_sdk import Cursor

for model in Cursor.models.list():
    print(model.id)
PY
```

Pass one of the returned IDs explicitly with `--model`.

## Analyze a model

From the repository root:

```bash
python tools/mcp/examples/run_with_cursor.py \
  "Analyze the add model with min_block_size=1. Only run analyze; do not compile, validate, or benchmark." \
  --model composer-2.5
```

Replace `composer-2.5` if it is not in the model list for the current account.

## Run a complete optimization workflow

The Agent decides which MCP tools to call from the task. State every required
stage and setting explicitly:

```bash
python tools/mcp/examples/run_with_cursor.py \
  "Compile the add model with min_block_size=1, validate the compiled artifact, and benchmark it with 10 warmup iterations and 100 measured iterations." \
  --model composer-2.5
```

For a multiline task, read the prompt from standard input:

```bash
cat <<'EOF' | python tools/mcp/examples/run_with_cursor.py - --model composer-2.5
Analyze the add model with min_block_size=1.
Only run analyze.
Report the job ID, node coverage, and partition assignment.
EOF
```

## Outputs

The command prints the isolated SDK run directory before starting:

```text
MCP artifacts: .../.mcp-runs/YYYYMMDD_HHMMSS_<8-character-id>
```

At completion it prints the SDK run status and runtime-reported token usage:

```text
Token usage
Run ID: run-...
Status: finished
Input: ...
Output: ...
Cache read: ...
Cache write: ...
Reasoning: ...
Total: ...
```

Every run writes `run.json` in that directory. It records the task, Cursor
agent and run IDs, selected model, timestamps, duration, status, errors, token
usage, and summaries of all MCP jobs created by the run. Pass
`--usage-json PATH` only when an additional standalone status and token-usage
file is needed.

The run directory also contains `effective-admin-config.json`, the source
administrator configuration with its `output_root` resolved for that run.

Token numbers cover the Cursor Agent run, including its prompts, reasoning,
MCP tool orchestration, and final response. They do not measure TensorRT
computation and do not include the separate Cursor IDE conversation.

If the run fails before usage is reported, `token_usage` in `run.json` is
`null`; this does not mean that the run used zero tokens.

## Command-line options

```text
task                 Optimization request, or "-" to read stdin
--model ID           Cursor model ID (default: auto)
--mcp-config PATH    MCP JSON file (default: .cursor/mcp.json)
--mcp-server NAME    MCP server entry (default: torch-tensorrt)
--output-root PATH   Override the generated MCP artifact directory
--usage-json PATH    Write an additional status and token usage JSON file
```

The generated or explicitly supplied output root must not already exist.

## Unsupported operator decomposition demo

The `diff` example exports `torch.diff` as the currently unsupported
`aten.diff.default` operator. It can be used to let an Agent generate and test
a targeted `slice + sub` decomposition:

```bash
python tools/mcp/examples/run_with_cursor.py \
  "Use torch-tensorrt MCP to compile, validate, and benchmark the diff model with min_block_size=1. Generate a custom decomposition for aten.diff.default if needed." \
  --model composer-2.5
```

Output:
```text
MCP artifacts: <repo>/.mcp-runs/20261112_001024_5af1cbde
Running the bounded optimization workflow: discovering MCP tools and locating
the diff model configuration.

Creating the decomposition candidate and running the candidate pipeline.
Decomposition enabled TensorRT but compiled latency regressed (~40×). Removing
the candidate per workflow criteria.

## Optimization result

Added a custom `aten.diff.default` decomposition (`slice` + `sub`) so the diff
model can lower into a TensorRT partition. Validation passed and an engine was
built. Compiled wall median was much worse than baseline (PyTorch fallback), so
the candidate source was removed per the retention rule.

Request settings:
  model / inputs: diff / diff
  model_parameters: {} (default)
  input_parameters: {} (batch 32, width 4096)
  seed: 0
  compilation: min_block_size=1 (plus MCP defaults)

Benchmark protocol (baseline and candidate):
  warmup: 10
  measured iterations: 100
  sample_index: 0
  timing scope: forward_only
  synchronization: per_invocation
  sample: batch-32-width-4096
  input: static fp16 [32, 4096]
  output: fp16 [32, 4095]

Baseline (no graph_transformation_candidate_id):
  analyze:   819f36772a9249058331a84b25d9ee31
  compile:   f7e21315c1d74794bc54b381035c0f7f
  validate:  23809b16102b4b5293a5dbdb6966b5aa
  benchmark: 456d6cd1146b4696a5d3b371de96eadd
  lowered graph: one aten.diff.default assigned to PyTorch
  TensorRT partitions: 0
  engines: 0
  validation: passed (atol=0.01, rtol=0.01, max error=0)
  compiled wall median: 0.0241125 ms
  benchmark wall speedup: 0.7477

Candidate decomposition:
  ID (removed): 29e485f2c9fe1cb46982d38ed73566a3
  source (deleted): <run-dir>/.graph_transformation_candidates/
    graph_transformation_candidate_29e485f2c9fe1cb46982d38ed73566a3.py
  analyze:   b233c9ca1f3b4306b91724d0aa07c0ff
  compile:   792c57f5e8b8465fb3a668534a7c2caf
  validate:  86a272ff1ce34e379d1f7811d8158e36
  benchmark: effaf01d29ed4281a0444ab981e36377
  transformation: aten.diff.default to 2 x aten.slice.Tensor + aten.sub.Tensor
  TensorRT coverage: 3/3 eligible, 0 unsupported
  engines: 1
  validation: passed with zero mismatches
  compiled wall median: 0.973067 ms
  benchmark wall speedup: 0.0205

Baseline-relative compiled speedup:
  0.0241125 / 0.973067 = 0.025x

The TensorRT path was about 40x slower than the PyTorch-fallback baseline for
this workload. The lowered graph changed as intended and validation passed, but
the candidate file was deleted because latency did not improve. All eight jobs
completed successfully.

Token usage:
  run ID: run-792cb9d4-cbd5-4106-a224-0a97c84ff1bf
  status: finished
  input: 903,784
  output: 7,337
  cache read: 853,120
  cache write: 0
  total: 911,121
Run summary: <repo>/.mcp-runs/20261112_001024_5af1cbde/run.json
```

TensorRT compilation and validation succeeded, but this graph is too small to
amortize runtime overhead. The example demonstrates custom decomposition and
end-to-end compilation rather than a performance gain over PyTorch eager.

## Batched linear optimization demo

The `batched_linear` example applies the same shared linear projection to eight
independent inputs. This prompt lets the Agent generate and evaluate one graph
transformation:

```bash
python tools/mcp/examples/run_with_cursor.py \
  "Use torch-tensorrt MCP to benchmark batched_linear_model. Requirements: min_block_size=1. The agent can optimize graphs to reduce latency." \
  --model composer-2.5
```

Output:
```text
MCP artifacts: <repo>/.mcp-runs/20261112_001752_67490569
Running the bounded graph-transformation workflow for batched_linear_model with
min_block_size=1. Discovering MCP tools and workspace context.

Starting baseline analyze and compile jobs.
Creating the graph transformation candidate and running the candidate job
pipeline.

## Optimization result

Applied a lowering pass that batches eight shared-weight aten.linear branches
into cat → one linear → eight slices, so TensorRT runs one large GEMM instead
of eight separate ones.

Request settings:
  model / inputs: batched_linear / batched_linear
  model_parameters: {} (default hidden_size=4096)
  input_parameters: {} (default batch_size=16)
  seed: 0
  compilation: min_block_size=1 plus MCP defaults

Benchmark protocol (baseline and candidate):
  warmup: 10
  iterations: 100
  sample_index: 0
  timing scope: forward_only
  synchronization: per_invocation
  sample: branches-8-batch-16-hidden-4096
  validation: atol=0.01, rtol=0.01

Baseline jobs (no graph_transformation_candidate_id):
  analyze:   e7a74275e7c8464a978f63251a468889
  compile:   80852e5647904da5bc53148c93023d8d
  validate:  7a9e0840fd5242af99d57bdae4a58f34
  benchmark: 75b286a603b546a49b4cdcfa4aa233b7

Baseline:
  lowered graph: eight independent aten.linear.default nodes, one shared
    weight, all in partition _run_on_acc_0
  computational nodes: 8
  compiled wall median: 2.12215 ms
  validation: passed for all 8 outputs
  benchmark wall speedup: 0.07935

Candidate:
  ID: 8e17b534ec44ba7090477bcd3dac38ea
  source: <run-dir>/.graph_transformation_candidates/
    graph_transformation_candidate_8e17b534ec44ba7090477bcd3dac38ea.py
  analyze:   d2696a8eafce4f0c8350233b7d0bff13
  compile:   21f67355db3945d2ba5bb3ada94c7d0b
  validate:  d98f7c34a988426c9f07c74b111196ed
  benchmark: db36288e5c044ab08373add6d63e1525
  lowered graph: aten.cat → aten.linear → 8 × aten.slice.Tensor
  graph I/O: same 8 inputs, 8 outputs, and shared weight
  compiled wall median: 1.473453 ms
  validation: passed
  benchmark wall speedup: 0.11592

Baseline-relative speedup: 2.12215 / 1.473453 = 1.44×
Latency reduction: approximately 30.5%

The candidate source was retained because the lowered graph changed as
intended, validation passed, and compiled median latency clearly improved.

Failures: none across baseline or candidate jobs.
MCP run directory: <repo>/.mcp-runs/20261112_001752_67490569

Token usage
Run ID: run-e01627db-5bd1-42fd-9eb2-f273baf8b24c
Status: finished
Input: 1,076,406
Output: 7,958
Cache read: 1,023,168
Cache write: 0
Total: 1,084,364
Run summary: <repo>/.mcp-runs/20261112_001752_67490569/run.json
```

The `speedup["wall"]` value in a benchmark report compares eager PyTorch with
that compiled artifact. The Agent instead calculates the optimization speedup
by dividing the baseline compiled wall median by the candidate compiled wall
median.

## Troubleshooting

### `CURSOR_API_KEY is required`

Export the key in the same terminal that runs the script.

### `This Cursor Router Optimize For mode was disabled for your team`

The default `--model auto` is disabled. List available models and pass an
explicit model ID.

### `libnvinfer.so` or `libnvonnxparser.so` cannot be found

Update `env.LD_LIBRARY_PATH` in the selected MCP JSON file to include the
directory containing the matching TensorRT libraries, then start a new script
run. Retrying the same MCP job without changing its environment will fail
again.

### `No token usage was reported for this run`

Check the printed `Status`, `Error code`, and `Error` fields or inspect the
run's `run.json` file. A failed or rejected run may not have usage metadata.

### MCP worker failure

Inspect the isolated SDK run directory printed by the script, then open the
relevant MCP job directory. Completed MCP jobs retain `failure.json`,
`manifest.json`, events, and worker diagnostics such as
`diagnostics/stderr.log`.

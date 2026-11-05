# Optimize Torch-TensorRT models with Cursor

`run_torch_trt_mcp_with_cursor.py` starts a Cursor Local Agent, gives it access
to the configured Torch-TensorRT MCP server, streams the Agent's response, and
reports the token usage for that SDK run.

The script can ask the MCP server to analyze, compile, validate, or benchmark
models allowlisted in `admin-config.json`. Each invocation gets an isolated
run directory under `mcp_example/sdk-runs/` by default. Its name contains
the UTC start time and an eight-character random identifier, for example
`20261105_011204_e04a1d83`. This keeps runs sortable and unique without sharing
the MCP job-manager lock used by the Cursor IDE.

## Prerequisites

- Linux with a working Torch, TensorRT, and Torch-TensorRT environment.
- The Torch-TensorRT MCP adapter installed in that environment.
- Python 3.10 or newer.
- A Cursor API key from [Cursor Dashboard → API Keys](https://cursor.com/dashboard/api-keys).
- A valid stdio MCP entry in `.cursor/mcp.json`.

Install the Cursor SDK from the repository root:

```bash
python -m pip install -r mcp_example/requirements-cursor-sdk.txt
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
        "/absolute/path/to/mcp_example/admin-config.json"
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

Models and input factories must be allowlisted in `mcp_example/admin-config.json`.
The included example defines the model and input name `add`.

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
python mcp_example/run_torch_trt_mcp_with_cursor.py \
  "Analyze the add model with min_block_size=1. Only run analyze; do not compile, validate, or benchmark." \
  --model composer-2.5
```

Replace `composer-2.5` if it is not in the model list for the current account.

## Run a complete optimization workflow

The Agent decides which MCP tools to call from the task. State every required
stage and setting explicitly:

```bash
python mcp_example/run_torch_trt_mcp_with_cursor.py \
  "Compile the add model with min_block_size=1, validate the compiled artifact, and benchmark it with 10 warmup iterations and 100 measured iterations." \
  --model composer-2.5
```

For a multiline task, read the prompt from standard input:

```bash
cat <<'EOF' | python mcp_example/run_torch_trt_mcp_with_cursor.py - --model composer-2.5
Analyze the add model with min_block_size=1.
Only run analyze.
Report the job ID, node coverage, and partition assignment.
EOF
```

## Outputs

The command prints the isolated SDK run directory before starting:

```text
MCP artifacts: .../mcp_example/sdk-runs/YYYYMMDD_HHMMSS_<8-character-id>
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

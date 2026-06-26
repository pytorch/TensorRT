# Contribution Guidelines

### Developing Torch-TensorRT

Do try to fill an issue with your feature or bug before filling a PR (op support is generally an exception as long as you provide tests to prove functionality). There is also a backlog (https://github.com/pytorch/TensorRT/issues) of issues which are tagged with the area of focus, a coarse priority level and whether the issue may be accessible to new contributors. Let us know if you are interested in working on a issue. We are happy to provide guidance and mentorship for new contributors. Though note, there is no claiming of issues, we prefer getting working code quickly vs. addressing concerns about "wasted work".

#### Development environment

Our build system relies on `bazel` (https://bazel.build/). Though there are many ways to install `bazel`, the preferred method is to use `bazelisk` (https://github.com/bazelbuild/bazelisk) which makes it simple to set up the correct version of bazel on the fly. Additional development dependencies can be installed via the `requirements-dev.txt` file.

#### Editor / clangd setup (optional)

For C++ code intelligence (go-to-definition, accurate diagnostics, completions) in any clangd-based editor — e.g. VSCode with the [clangd extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd), Cursor, Neovim, Emacs — generate a Bazel-aware compilation database:

```sh
bazel run //:refresh_compile_commands
```

This writes `compile_commands.json` at the workspace root (gitignored). Re-run it after pulling changes that materially affect the build graph (new targets, new headers, dependency bumps). The repo's `.clangd` file picks up the database automatically.

This is opt-in: developers who don't use clangd are unaffected, and the underlying `hedron_compile_commands` extractor is declared as a Bazel `dev_dependency`, so it does not enter the dep graph of consumers building against torch_tensorrt.

#### Communication

The primary location for discussion is GitHub issues and Github discussions. This is the best place for questions about the project and discussion about specific issues.

We use the PyTorch Slack for communication about core development, integration with PyTorch and other communication that doesn't make sense in GitHub issues. If you need an invite, take a look at the [PyTorch README](https://github.com/pytorch/pytorch/blob/master/README.md) for instructions on requesting one.

### Coding Guidelines

- We generally follow the coding guidelines used in PyTorch

    - Linting your code is essential to ensure code matches the style guidelines.
      To begin with, please install the following dependencies
      * `pip install -r requirements-dev.txt`
      * Install Bazel buildifier https://github.com/bazelbuild/buildtools/blob/master/buildifier/README.md#setup

      Once the above dependencies are installed, `git commit` command will perform linting before committing your code.

- Avoid introducing unnecessary complexity into existing code so that maintainability and readability are preserved

- Try to avoid committing commented out code

- Minimize warnings (and no errors) from the compiler

- Make sure all converter tests and the core module testsuite pass

- New features should have corresponding tests or if its a difficult feature to test in a testing framework, your methodology for testing.

- Comment subtleties and design decisions

- Document hacks, we can discuss it only if we can find it

### Controlling CI scope via PR labels

**PR default:** build + L0 + L1 on a single representative config (Python 3.12 × CUDA 13.0). L2 (slow model-level suites) is opt-in on PRs. The full matrix ({Python 3.10–3.13} × {CUDA 13.0, 13.2}) runs on main / nightly / release branches.

Apply labels in the PR's right sidebar and re-push (or close/reopen) to re-trigger:

| Label | Effect |
|---|---|
| `ci: only-l0` | Skip L1 and L2. Useful for docs / build-system-only changes. |
| `ci: run-l2` | Opt-in to L2 model-compilation tests on this PR. |
| `ci: skip-l2` | (Legacy — no-op on PRs since L2 is now off by default; still skips L2 on main-branch push runs.) |
| `Force All Tests[L0+L1+L2]` | Force every tier to run even if an earlier tier failed. Also enables L2 on PRs. |

The Linux x86_64 standard and RTX pipelines share a single reusable workflow,
`.github/workflows/_linux-x86_64-core.yml`. The two entry workflows
(`build-test-linux-x86_64.yml` and `build-test-linux-x86_64_rtx.yml`) just call
it with `use-rtx: false` / `true` and render a single rollup check
(`CI / Linux x86_64` and `CI / Linux x86_64 (RTX)`). Edit test scope, tiers, or
gating in the core — not the entry workflows. RTX-only differences are gated
with `if: ${{ !inputs.use-rtx ... }}` (for standard-only jobs) or branch inside
the script on the `$USE_TRT_RTX` env var (for divergent test scope).

### Commits and PRs

- Try to keep pull requests focused (multiple pull requests are okay). Typically PRs should focus on a single issue or a small collection of closely related issue.

- Typically we try to follow the guidelines set by https://www.conventionalcommits.org/en/v1.0.0/ for commit messages for clarity. Again not strictly enforced.

- We require that all contributors sign CLA for submitting PRs. In order for us to review and merge your suggested changes, please sign at https://code.facebook.com/cla. If you are contributing on behalf of someone else (eg your employer), the individual CLA may not be sufficient and your employer may need to sign the corporate CLA.

- We have Git hooks set up to perform common checks and pre-commit tasks such as linting for Python, C++ and Bazel files. In order to use these tools please install `pre-commit` as well as `buildifier`

```sh
pip install pre-commit
go install github.com/bazelbuild/buildtools/buildifier@latest
```

## Local testing

Once you have a built/installed Torch-TensorRT (see the build docs), use the
[`just`](https://github.com/casey/just) recipes in the repo root to run the
same checks CI runs, against your local checkout. They drive `uv` with
`--no-sync` (so they use your already-built environment instead of rebuilding
from source) and isolate the engine/timing cache under a per-user `$TMPDIR`.

```sh
just                 # list all recipes
just lint            # run every pre-commit hook (matches the linter CI job)
just lint-changed    # pre-commit on files changed vs origin/main (fast pre-push)
just test <args>     # pytest in the uv env, e.g. `just test tests/py/dynamo/conversion/`

# Reproduce a whole CI tier before pushing (selectors mirror _linux-x86_64-core.yml):
just l0              # full L0 smoke tier (converter + core + py-core + torchscript)
just l1              # full L1 tier
just l0-converter    # or a single sub-suite
```

`just test` and the tier recipes accept the same args/flags as pytest. On a
single local GPU, the default `-n auto` parallelism may exceed GPU memory when
many TRT engines build at once — lower it with `just jobs=2 l0`.

> Legacy: `noxfile.py` still defines multi-Python-version sessions
> (`nox --session -l`) used by some release flows, but `just` is the
> recommended path for everyday local verification.

## How do I add support for a new op...

### In Torch-TensorRT?

Thanks for wanting to contribute! There are two main ways to handle supporting a new op. Either you can write a converter for the op from scratch and register it in the NodeConverterRegistry or if you can map the op to a set of ops that already have converters you can write a graph rewrite pass which will replace your new op with an equivalent subgraph of supported ops. Its preferred to use graph rewriting because then we do not need to maintain a large library of op converters. Also do look at the various op support trackers in the [issues](https://github.com/pytorch/TensorRT/issues) for information on the support status of various operators.

### In my application?

> The Node Converter Registry is not exposed in the top level API but in the internal headers shipped with the tarball.

You can register a converter for your op using the `NodeConverterRegistry` inside your application.

## Structure of the repo

| Component                | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| [**core**](core)         | Main JIT ingest, lowering, conversion and runtime implementations |
| [**cpp**](cpp)           | C++ API and CLI source                                       |
| [**examples**](examples) | Example applications to show different features of Torch-TensorRT |
| [**py**](py)             | Python API for Torch-TensorRT                                |
| [**tests**](tests)       | Unit tests for Torch-TensorRT                                |

Thanks in advance for your patience as we review your contributions; we do appreciate them!
# Contribution Guidelines

### Developing Torch-TensorRT

Do try to fill an issue with your feature or bug before filling a PR (op support is generally an exception as long as you provide tests to prove functionality). There is also a backlog (https://github.com/pytorch/TensorRT/issues) of issues which are tagged with the area of focus, a coarse priority level and whether the issue may be accessible to new contributors. Let us know if you are interested in working on a issue. We are happy to provide guidance and mentorship for new contributors. Though note, there is no claiming of issues, we prefer getting working code quickly vs. addressing concerns about "wasted work".

#### Development environment

Our build system relies on `bazel` (https://bazel.build/). Though there are many ways to install `bazel`, the preferred method is to use `bazelisk` (https://github.com/bazelbuild/bazelisk) which makes it simple to set up the correct version of bazel on the fly. Additional development dependencies can be installed via the `requirements-dev.txt` file.

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

### Commits and PRs

- Try to keep pull requests focused (multiple pull requests are okay). Typically PRs should focus on a single issue or a small collection of closely related issue.

- Typically we try to follow the guidelines set by https://www.conventionalcommits.org/en/v1.0.0/ for commit messages for clarity. Again not strictly enforced.

- We require that all contributors sign CLA for submitting PRs. In order for us to review and merge your suggested changes, please sign at https://code.facebook.com/cla. If you are contributing on behalf of someone else (eg your employer), the individual CLA may not be sufficient and your employer may need to sign the corporate CLA.

- We have Git hooks set up to perform common checks and pre-commit tasks such as linting for Python, C++ and Bazel files. In order to use these tools please install `pre-commit` as well as `buildifier`

```sh
pip install pre-commit
go install github.com/bazelbuild/buildtools/buildifier@latest
```

## Testing using Python backend

Torch-TensorRT supports testing in Python using [nox](https://nox.thea.codes/en/stable)

To install the nox using python-pip

```
python3 -m pip install --upgrade nox
```

To list supported nox sessions:

```
nox --session -l
```

Environment variables supported by nox

```
PYT_PATH          - To use different PYTHONPATH than system installed Python packages
TOP_DIR           - To set the root directory of the noxfile
USE_HOST_DEPS     - To use host dependencies for tests (Defaults to 0)
```

Usage example

```
nox --session l0_api_tests
```

Supported Python versions:
```
["3.7", "3.8", "3.9", "3.10"]
```

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
# Tests

The Python tests cover the Dynamo frontend, TensorRT conversion, partitioning,
lowering, serialization and runtime behavior. The CI suite manifest is
[`ci/suites.py`](ci/suites.py).

```sh
python -m tests.ci list
python -m pytest tests/py/core
python -m pytest tests/py/dynamo
```

Native runtime and ExecuTorch backend tests are under `core/runtime` and
`cpp/executorch`. Run them with `bazel test //tests --test_output=errors`.
The legacy TorchScript compiler, converter, model and serialization suites
have been removed.

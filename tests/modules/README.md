# Module Level Tests

These tests focus on integration testing of the backend, using the public apis. They require TorchScript modules saved to files. You can download the set of modules used by default in the tests by running hub.py (you need PyTorch 1.9.0 installed with cuda-11.1).

## Running Tests

``` shell
bazel test //tests/modules:test_modules --compilation_mode=dbg --test_output=errors
```

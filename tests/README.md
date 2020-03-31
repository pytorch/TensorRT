# Tests

Right now there are two types of tests. Converter level tests and Module level tests.

The goal of Converter tests are to tests individual converters againsts specific subgraphs. The current tests in `core/conveters` are good examples on how to write these tests. In general every converter should have at least 1 test. More may be required if the operation has switches that change the behavior of the op.

Module tests are designed to test the compiler against common network architectures and verify the integration of converters together into a single engine.

You can run the whole test suite with bazel. But be aware you may exhaust GPU memory (this may be seen as a cuDNN initialization error) running them naively, you therefore may need to limit the number of concurrent tests. Also because the inputs to tests are random it may make sense to run tests a few times.

Here are some settings that work well the current test suite on a TITAN V.

```
bazel test //tests --compilation_mode=dbg --test_output=errors --jobs=4 --runs_per_test=5
```

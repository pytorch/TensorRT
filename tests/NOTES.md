# Tests

## Lists of Tests disabled for DLFW CI

Following tests have been disabled due to flaky output issues with DLFW integration CI. These test cases will remain disabled until fixed for integration with DLFW. 

```
1. MobileNet_v2 model test (test_api.py)
2. GELU test (test_activation.cpp)
3. LSTM tests (test_lstm_cell.cpp)
4. Softmax tests (test_softmax.cpp) 
5. Partitioning tests (test_fallback_graph_output.cpp)
6. RuntimeThreadSafety tests (test_runtime_thread_safety.cpp)
7. ModuleAsEngine tests (test_modules_as_engines.cpp)
8. ModuleFallback tests (test_module_fallback.cpp)
9. MultipleRunEngine tests (test_multiple_registered_engines.cpp)
10. CompiledModule tests (test_compiled_modules.cpp)

```

Note: Most of these tests could be identified with DISABLE_TEST_IN_CI flag in the test suite.

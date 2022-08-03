# Tests

## Lists of Tests disabled for DLFW CI

Following tests have been disabled due to flaky output issues with DLFW integration CI. These test cases will remain disabled until fixed for integration with DLFW.

```
1. MobileNet_v2 model TestCompile test (test_api.py)

2. ATenGELUConvertsCorrectly test (test_activation.cpp)

3. LSTM tests (test_lstm_cell.cpp)
   a. ATenGRUCellConvertsCorrectlyWithBiasCheckHidden
   b. ATenGRUCellConvertsCorrectlyWithoutBiasCheckHidden
   c. ATenLSTMCellConvertsCorrectlyWithBiasCheckHidden
   d. ATenLSTMCellConvertsCorrectlyWithBiasCheckCell
   e. ATenLSTMCellConvertsCorrectlyWithoutBiasCheckHidden
   f. ATenLSTMCellConvertsCorrectlyWithoutBiasCheckCell

4. Softmax tests (test_softmax.cpp)
   a. ATenSoftmax1DConvertsCorrectly
   b. ATenSoftmaxNDConvertsCorrectlySub3DIndex
   c. ATenSoftmaxNDConvertsCorrectlyAbove3DIndex
   d. ATenSoftmaxNDConvertsCorrectlyNegtiveOneIndex
   e. ATenSoftmaxNDConvertsCorrectlyNegtiveIndex

5. Partitioning tests (test_fallback_graph_output.cpp)
   a. ComputeResNet50FallbackGraphCorrectly
   b. ComputeMobileNetFallbackGraphCorrectly
   c. ComputeResNet50HalfFallbackGraphCorrectly

6. RuntimeThreadSafety tests (test_runtime_thread_safety.cpp)

7. ModuleAsEngine tests (test_modules_as_engines.cpp)
   a. ModuleAsEngineIsClose
   b. ModuleToEngineToModuleIsClose

8. ModuleFallback tests (test_module_fallback.cpp)
   a. ResNetModuleFallbacksCorrectly
   b. MobileNetModuleFallbacksCorrectlyWithOneEngine

9. CanRunMultipleEngines tests (test_multiple_registered_engines.cpp)

10. CompiledModuleIsClose tests (test_compiled_modules.cpp)
```

Note: Most of these tests could be identified with DISABLE_TEST_IN_CI flag in the test suite.

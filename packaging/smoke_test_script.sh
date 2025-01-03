# Smoke test is intentionally disabled.
# The issue was smoke test installs the built torch_tensorrt wheel file and checks `import torch_tensorrt; print(torch_tensorrt.__version__)`
# Since tensorrt cannot be pip installable in CI, the smoke test will fail.
# One way we tried to handle it is manually install tensorrt wheel while by extracting from the tarball.
# However, the TensorRT-10.7.0.23/lib path doesn't seem to show up in LD_LIBRARY_PATH even if we explicitly set it.
# TODO: Implement a custom smoke_test script to verify torch_tensorrt installation.
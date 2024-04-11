python -m pip install -U numpy packaging pyyaml setuptools wheel
python -c "import torch; print('IMPORTED TORCH 1')"
python -m pip config set global.index-url "https://pypi.org/simple https://download.pytorch.org/whl/nightly/cu118 https://download.pytorch.org/whl/nightly/cu121"
python -m pip config set global.extra-index-url "https://pypi.nvidia.com"
python -m pip config set global.no-cache-dir true
python -c "import torch; print('IMPORTED TORCH 2')"
python -m pip install tensorrt==9.3.0.post12.dev1 tensorrt_libs==9.3.0.post12.dev1 tensorrt_bindings==9.3.0.post12.dev1
python -c "import torch; print('IMPORTED TORCH 3')"

choco install bazelisk -y

cat ./toolchains/ci_workspaces/WORKSPACE.win.release.tmpl | envsubst > WORKSPACE

# Adapted from:
# https://github.com/HolyWu/TensorRT/commit/41237cedd56809062ad8e29ed0a18fabd54dd3e1#diff-97ebcd21a0f2f4e262f4e6c703e13e8cc6b233f9d9861eb2d9d06994f555f144
curl -so cuda_12.1.0_windows_network.exe https://developer.download.nvidia.com/compute/cuda/12.1.0/network_installers/cuda_12.1.0_windows_network.exe
./cuda_12.1.0_windows_network.exe -s nvcc_12.1 Display.Driver -n
# ./cuda_12.1.0_windows_network.exe -s cuda_profiler_api_12.1 cudart_12.1 cuobjdump_12.1 cupti_12.1 cuxxfilt_12.1 nvcc_12.1 nvdisasm_12.1 nvjitlink_12.1 nvml_dev_12.1 nvprof_12.1 nvprune_12.1 nvrtc_12.1 nvrtc_dev_12.1 nvtx_12.1 nvvm_samples_12.1 opencl_12.1 visual_profiler_12.1 sanitizer_12.1 thrust_12.1 cublas_12.1 cublas_dev_12.1 cufft_12.1 cufft_dev_12.1 curand_12.1 curand_dev_12.1 cusolver_12.1 cusolver_dev_12.1 cusparse_12.1 cusparse_dev_12.1 npp_12.1 npp_dev_12.1 nvjpeg_12.1 nvjpeg_dev_12.1 -n

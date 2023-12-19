# @REM powershell {$ProgressPreference="SilentlyContinue"; Invoke-WebRequest https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/zip/TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0.zip -OutFile .\TensorRT-8.6.1.6.zip; Expand-Archive .\TensorRT-8.6.1.6.zip -DestinationPath .\TensorRT-8.6.1.6; $trt_lib=$pwd.Path+'\TensorRT-8.6.1.6\lib'; $userPath=[Environment]::GetEnvironmentVariable('Path', 'User'); [Environment]::SetEnvironmentVariable('Path', ($userPath + ';' + $trt_lib), 'User'); $env:Path+=';'+$trt_lib; python -m pip install $pwd+'\TensorRT-8.6.1.6\TensorRT-8.6.1.6\python\tensorrt-8.6.1-cp310-none-win_amd64.whl'}
python -m pip install pyyaml packaging numpy
python -c "import torch; print('IMPORTED TORCH 1')"
python -m pip config set global.index-url "https://pypi.org/simple https://download.pytorch.org/whl/nightly/cu118 https://download.pytorch.org/whl/nightly/cu121"
python -m pip config set global.extra-index-url "https://pypi.nvidia.com"
python -m pip config set global.no-cache-dir true
python -c "import torch; print('IMPORTED TORCH 2')"
python -m pip install --no-deps tensorrt==9.2.0.post12.dev5 tensorrt_libs==9.2.0.post12.dev5 tensorrt_bindings==9.2.0.post12.dev5
python -c "import torch; print('IMPORTED TORCH 3')"

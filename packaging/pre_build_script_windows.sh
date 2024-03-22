python -m pip install pyyaml packaging numpy
python -c "import torch; print('IMPORTED TORCH 1')"
python -m pip config set global.index-url "https://pypi.org/simple https://download.pytorch.org/whl/nightly/cu118 https://download.pytorch.org/whl/nightly/cu121"
python -m pip config set global.extra-index-url "https://pypi.nvidia.com"
python -m pip config set global.no-cache-dir true
python -c "import torch; print('IMPORTED TORCH 2')"
python -m pip install tensorrt==9.3.0.post12.dev1 tensorrt_libs==9.3.0.post12.dev1 tensorrt_bindings==9.3.0.post12.dev1
python -c "import torch; print('IMPORTED TORCH 3')"
choco install bazelisk -y
cat toolchains/ci_workspaces/WORKSPACE.win.release.tmpl | envsubst > WORKSPACE

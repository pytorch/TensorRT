python -m pip install -U numpy packaging pyyaml setuptools wheel

# Install TRT 10 from PyPi
python -m pip install tensorrt==10.0.1 --extra-index-url https://pypi.nvidia.com

choco install bazelisk -y

cat toolchains/ci_workspaces/WORKSPACE.win.release.tmpl | envsubst > WORKSPACE

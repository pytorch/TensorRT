python -m pip install -U numpy packaging pyyaml setuptools wheel

# Install TRT 10 from PyPi
TRT_VERSION=$(python -c "import yaml; print(yaml.safe_load(open('dev_dep_versions.yml', 'r'))['__tensorrt_version__'])")
pip install tensorrt==${TRT_VERSION} tensorrt-cu12-bindings==${TRT_VERSION} tensorrt-cu12-libs==${TRT_VERSION} --extra-index-url https://pypi.nvidia.com

choco install bazelisk -y

cat toolchains/ci_workspaces/WORKSPACE.win.release.tmpl | envsubst > WORKSPACE

echo "RELEASE=1" >> ${GITHUB_ENV}

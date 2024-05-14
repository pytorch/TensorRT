python -m pip install -U numpy packaging pyyaml setuptools wheel

# Install TRT from PyPi
TRT_VERSION=$(${CONDA_RUN} python -c "import yaml; print(yaml.safe_load(open('dev_dep_versions.yml', 'r'))['__tensorrt_version__'])")

python -m pip install tensorrt==${TRT_VERSION} tensorrt-${CU_VERSION::4}==${TRT_VERSION} tensorrt-${CU_VERSION::4}-bindings==${TRT_VERSION} tensorrt-${CU_VERSION::4}-libs==${TRT_VERSION} --extra-index-url https://pypi.nvidia.com

choco install bazelisk -y

cat toolchains/ci_workspaces/WORKSPACE.win.release.tmpl | envsubst > WORKSPACE

echo "RELEASE=1" >> ${GITHUB_ENV}

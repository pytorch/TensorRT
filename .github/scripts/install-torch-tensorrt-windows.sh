set -eou pipefail
source "${BUILD_ENV_FILE}"

# Install pyyaml first to parse dev_dep_versions.yml in versions.py
${CONDA_RUN} pip install pyyaml

# Install stable versions of Torch and Torchvision since this is a release branch.
# For main branches, the INDEX_URL will always point to nightly and for RC releases,
# it will contain test tag in the url.
export TORCH_VERSION_CI=$(${CONDA_RUN} python -c "import versions; versions.torch_version()")
export TORCHVISION_VERSION_CI=$(${CONDA_RUN} python -c "import versions; versions.torchvision_version()")
export TENSORRT_VERSION_CI=$(${CONDA_RUN} python -c "import versions; versions.tensorrt_version()")
export INDEX_URL_CI=$(${CONDA_RUN} python -c "import versions; versions.index_url()")

# Install appropriate torch and torchvision versions for the platform
${CONDA_RUN} pip install torch==${TORCH_VERSION_CI} torchvision==${TORCHVISION_VERSION_CI} --index-url ${INDEX_URL_CI}

# Install TRT 10 from PyPi
${CONDA_RUN} pip install tensorrt==${TENSORRT_VERSION_CI}

# Install pre-built Torch-TRT
${CONDA_RUN} pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl

echo -e "Running test script";

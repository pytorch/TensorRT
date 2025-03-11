#!/bin/bash

check_command() {
    command -v "$1" >/dev/null 2>&1
}

ensure_installed() {
    local pkg="$1"
    if ! check_command "$pkg"; then
        echo "$pkg is not installed. Installing $pkg..."

        # Determine if sudo is needed
        if check_command sudo; then
            SUDO="sudo"
        else
            SUDO=""
        fi

        # Detect OS and install accordingly
        OS="$(uname -s)"
        if [[ "$OS" == "Linux" ]]; then
            if check_command apt-get; then
                $SUDO apt-get update && $SUDO apt-get install -y "$pkg"
            fi
        else
            echo "Unsupported OS: $OS. Please install $pkg manually."
            exit 1
        fi
    else
        echo "$pkg is already installed."
    fi
}

ensure_mpi_installed() {
    local pkg="$1"
    if dpkg -l | grep -q "$pkg"; then
        echo "$pkg is already installed."
    else
        echo "$pkg is not installed. Installing $pkg..."

        # Determine if sudo is needed
        if check_command sudo; then
            SUDO="sudo"
        else
            SUDO=""
        fi

        # Detect OS and install accordingly
        OS="$(uname -s)"
        if [[ "$OS" == "Linux" ]]; then
            if check_command apt-get; then
                $SUDO apt-get update && $SUDO apt-get install -y "$pkg"
            fi
        else
            echo "Unsupported OS: $OS. Please install $pkg manually."
            exit 1
        fi
    fi
}

ensure_pytest_installed(){
    if check_command pip; then
        echo "pip is installed, installing pytest..."
        pip install pytest
    else
        echo "pip is not installed. Please install pip first."
        exit 1
    fi
}

echo "Setting up the environment"

OS="$(uname -s)"
ARCH="$(uname -m)"


#getting the file name for TensorRT-LLM download
if [[ "$OS" == "Linux" && "$ARCH" == "x86_64"]]; then
    FILE="tensorrt_llm-0.17.0.post1-cp312-cp312-linux_x86_64.whl"
elif [[ "$OS" == "Linux" && "$ARCH" == "aarch64"]]; then
    FILE="tensorrt_llm-0.17.0.post1-cp312-cp312-linux_aarch64.whl"
else:
    echo "Unsupported platform: OS=$OS ARCH=$ARCH
    exit 1
fi

# Download the selected file
URL="https://pypi.nvidia.com/tensorrt-llm/$FILE"
echo "Downloading $FILE from $URL..."

echo "Downloading here...."
#Installing wget
ensure_installed wget

#Downloading the file
filename=$(basename "$URL")
if [ -f "$filename" ]; then
    echo "File already exists: $filename"
else
    wget "$URL"
fi
echo "Download complete: $FILE"

UNZIP_DIR="tensorrt_llm_unzip"
if [[ ! -d "$UNZIP_DIR" ]]; then
    echo "Creating directory: $UNZIP_DIR"
    mkdir -p "$UNZIP_DIR"
    echo "extracting $FILE to $UNZIP_DIR ..."
    #Installing unzip
    ensure_installed unzip
    #unzip the TensorRT-LLM package
    unzip -q "$FILE" -d "$UNZIP_DIR"
    echo "Unzip complete"
fi


export TRTLLM_PLUGINS_PATH="$(pwd)/${UNZIP_DIR}/tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so"
echo ${TRTLLM_PLUGINS_PATH}

ensure_mpi_installed libmpich-dev
ensure_mpi_installed libopenmpi-dev

run_tests() {
    cd ..
    export PYTHONPATH=$(pwd)
    echo "Running pytest on distributed/test_nccl_ops.py..."
    pytest distributed/test_nccl_ops.py
}

run_mpi_tests(){
    cd distributed
    echo "Running test_distributed_simple_example with mpirun..."---
    mpirun -n 1 --allow-run-as-root python test_distributed_simple_example.py
}

ensure_pytest_installed
run_tests
run_mpi_tests
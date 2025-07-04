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


ensure_mpi_installed libmpich-dev
ensure_mpi_installed libopenmpi-dev

run_tests() {
    cd ..
    export PYTHONPATH=$(pwd)
    echo "Running pytest on distributed/test_nccl_ops.py..."
    USE_TRTLLM_PLUGINS=1 pytest distributed/test_nccl_ops.py
}

run_mpi_tests(){
    cd distributed
    echo "Running test_distributed_simple_example with mpirun..."---
    mpirun -n 1 --allow-run-as-root python test_distributed_simple_example.py
}

ensure_pytest_installed
run_tests
run_mpi_tests
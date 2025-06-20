
install_cuda_aarch64() {
    echo "install cuda ${CU_VERSION}"
    # CU_VERSION: cu128 --> CU_VER: 12-8
    CU_VER=${CU_VERSION:2:2}-${CU_VERSION:4:1}
    # CU_VERSION: cu128 --> CU_DOT_VER: 12.8
    CU_DOT_VER=${CU_VERSION:2:2}.${CU_VERSION:4:1}
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo
    
    nvidia_drivers=$(dnf list installed | grep nvidia-driver)
    echo "before install: dnf list installed | grep nvidia-driver: ${nvidia_drivers}"
    
    dnf -y install nvidia-driver nvidia-driver-cuda \
                   cuda-compiler-${CU_VER}.aarch64 \
                   cuda-libraries-${CU_VER}.aarch64 \
                   cuda-libraries-devel-${CU_VER}.aarch64 \
                   libnccl-2.26.5-1+cuda${CU_DOT_VER} libnccl-devel-2.26.5-1+cuda${CU_DOT_VER} libnccl-static-2.26.5-1+cuda${CU_DOT_VER}
    dnf clean all
    
    nvidia_drivers=$(dnf list installed | grep nvidia-driver)
    echo "after install: dnf list installed | grep nvidia-driver: ${nvidia_drivers}"
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:$LD_LIBRARY_PATH
    ls -lart /usr/local/
    nvcc --version
    # Check if nvidia-smi is available and working
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
    else
        echo “nvidia-smi not found - no NVIDIA GPU or drivers installed”
    fi
    # Check for NVIDIA device files in /dev
    if compgen -G “/dev/nvidia[0-9]” >/dev/null; then
        echo “NVIDIA GPU devices found:”
        ls -la /dev/nvidia*
    else
        echo “No NVIDIA GPU devices found in /dev”
    fi
    # Check for NVIDIA GPU controllers in PCI devices
    if lspci -v | grep -e ‘controller.*NVIDIA’ >/dev/null 2>/dev/null; then
        echo “NVIDIA GPU found”
        lspci | grep -i nvidia
    else
        echo “No NVIDIA GPU found”
    fi

    # Check if NVIDIA kernel module is loaded
    if lsmod | grep -q nvidia; then
        echo “NVIDIA kernel module is loaded”
        lsmod | grep nvidia
    else
        echo “NVIDIA kernel module not loaded”
    fi
    echo "cuda ${CU_VER} installed successfully"
}


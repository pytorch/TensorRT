
install_cuda_aarch64() {
    echo "install cuda${CU_VERSION}"
    # CU_VERSION: cu128 --> CU_VER: 12-8
    CU_VER=${CU_VERSION:2:2}-${CU_VERSION:4:1}
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo
    dnf -y install cuda-compiler-${CU_VER}.aarch64 \
                   cuda-libraries-${CU_VER}.aarch64 \
                   cuda-libraries-devel-${CU_VER}.aarch64
    dnf clean all
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    ls -lart /usr/local/
    nvcc --version
    echo "cuda ${CU_VER} installed successfully"
}


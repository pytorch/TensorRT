
install_cuda_aarch64() {
    echo "install cuda ${CU_VERSION}"
    # CU_VERSION: cu128 --> CU_VER: 12-8
    CU_VER=${CU_VERSION:2:2}-${CU_VERSION:4:1}
    # CU_VERSION: cu129 --> CU_DOT_VER: 12.9
    CU_DOT_VER=${CU_VERSION:2:2}.${CU_VERSION:4:1}
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo
    # nccl version must match libtorch_cuda.so was built with https://github.com/pytorch/pytorch/blob/main/.ci/docker/ci_commit_pins/nccl-cu12.txt
    dnf -y install cuda-compiler-${CU_VER}.aarch64 \
                   cuda-libraries-${CU_VER}.aarch64 \
                   cuda-libraries-devel-${CU_VER}.aarch64 \
                   libnccl-2.27.3-1+cuda${CU_DOT_VER} libnccl-devel-2.27.3-1+cuda${CU_DOT_VER} libnccl-static-2.27.3-1+cuda${CU_DOT_VER}
    dnf clean all
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:$LD_LIBRARY_PATH
    ls -lart /usr/local/
    nvcc --version
    echo "cuda ${CU_VER} installed successfully"
}



install_cuda_aarch64() {
    echo "install cuda ${CU_VERSION}"
    # CU_VERSION: cu128 --> CU_VER: 12-8
    CU_VER=${CU_VERSION:2:2}-${CU_VERSION:4:1}
    # CU_VERSION: cu129 --> CU_DOT_VER: 12.9
    CU_DOT_VER=${CU_VERSION:2:2}.${CU_VERSION:4:1}
    # CUDA_MAJOR_VERSION: cu128 --> 12
    CUDA_MAJOR_VERSION=${CU_VERSION:2:2}
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo

    # nccl version must match libtorch_cuda.so was built with
    if [[ ${CU_VERSION:0:4} == "cu12" ]]; then
        # cu12: https://github.com/pytorch/pytorch/blob/main/.ci/docker/ci_commit_pins/nccl-cu12.txt
        if [[ ${CU_VERSION} == "cu128" ]]; then
            nccl_version="2.26.2-1"
        elif [[ ${CU_VERSION} == "cu126" ]]; then
            nccl_version="2.24.3-1"
        else
            # removed cu129 support from pytorch upstream
            echo "Unsupported CUDA version: ${CU_VERSION}"
            exit 1
        fi
    elif [[ ${CU_VERSION:0:4} == "cu13" ]]; then
        # cu13: https://github.com/pytorch/pytorch/blob/main/.ci/docker/ci_commit_pins/nccl-cu13.txt
        nccl_version="2.27.7-1"
    fi

    dnf --nogpgcheck -y install cuda-compiler-${CU_VER}.aarch64 \
                   cuda-libraries-${CU_VER}.aarch64 \
                   cuda-libraries-devel-${CU_VER}.aarch64 \
                   libnccl-${nccl_version}+cuda${CU_DOT_VER} libnccl-devel-${nccl_version}+cuda${CU_DOT_VER} libnccl-static-${nccl_version}+cuda${CU_DOT_VER}
    dnf clean all
    # nvshmem version is from https://github.com/pytorch/pytorch/blob/f9fa138a3910bd1de1e7acb95265fa040672a952/.ci/docker/common/install_cuda.sh#L67
    nvshmem_version=3.3.24
    nvshmem_path="https://developer.download.nvidia.com/compute/redist/nvshmem/${nvshmem_version}/builds/cuda${CUDA_MAJOR_VERSION}/txz/agnostic/aarch64"
    nvshmem_prefix="libnvshmem-linux-sbsa-${nvshmem_version}_cuda${CUDA_MAJOR_VERSION}-archive"
    nvshmem_tarname="${nvshmem_prefix}.tar.xz"
    curl -L ${nvshmem_path}/${nvshmem_tarname} -o nvshmem.tar.xz
    tar -xJf nvshmem.tar.xz
    cp -a ${nvshmem_prefix}/lib/* /usr/local/cuda/lib64/
    cp -a ${nvshmem_prefix}/include/* /usr/local/cuda/include/
    rm -rf nvshmem.tar.xz ${nvshmem_prefix}
    echo "nvshmem ${nvshmem_version} for cuda ${CUDA_MAJOR_VERSION} installed successfully"

    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/include:/usr/lib64:$LD_LIBRARY_PATH
    ls -lart /usr/local/
    nvcc --version
    echo "cuda ${CU_VER} installed successfully"
}


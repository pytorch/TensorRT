# for now we only need to install cuda_dss for jetpack
install_cuda_dss_aarch64() {
    echo "install cuda_dss for ${CU_VERSION}"
    arch_path='sbsa'
    # version is from https://github.com/pytorch/pytorch/blob/22c5e8c17c7551c9dd2855589ae774c1e147343a/.ci/docker/common/install_cudss.sh
    CUDSS_NAME="libcudss-linux-${arch_path}-0.3.0.9_cuda12-archive"
    curl --retry 3 -OLs https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-${arch_path}/${CUDSS_NAME}.tar.xz
    # only for cuda 12
    tar xf ${CUDSS_NAME}.tar.xz
    cp -a ${CUDSS_NAME}/include/* /usr/local/cuda/include/
    cp -a ${CUDSS_NAME}/lib/* /usr/local/cuda/lib64/
}
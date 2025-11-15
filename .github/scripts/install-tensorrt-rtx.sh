
install_tensorrt_rtx() {
    if [[ ${USE_TRT_RTX} == true ]]; then
        if [[ ${CU_VERSION:2:2} == "13" ]]; then
            export CU_UPPERBOUND="13.0"
        else
            export CU_UPPERBOUND="12.9"
        fi
        TRT_RTX_VERSION=1.2.0.54
        install_wheel_or_not=${1:-false}
        echo "It is the tensorrt-rtx build, install tensorrt-rtx with install_wheel_or_not:${install_wheel_or_not}"
        PLATFORM=$(python -c "import sys; print(sys.platform)")
        echo "PLATFORM: $PLATFORM"
        # PYTHON_VERSION is always set in the CI environment, add this check for local testing
        if [ -z "$PYTHON_VERSION" ]; then
            echo "Error: PYTHON_VERSION environment variable is not set or empty. example format: export PYTHON_VERSION=3.11"
            exit 1
        fi

        # python version is like 3.11, we need to convert it to cp311
        CPYTHON_TAG="cp${PYTHON_VERSION//./}"
        if [[ ${PLATFORM} == win32 ]]; then
            curl -L https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.2/tensorrt-rtx-${TRT_RTX_VERSION}-win10-amd64-cuda-${CU_UPPERBOUND}-release-external.zip -o tensorrt-rtx-${TRT_RTX_VERSION}.win10-amd64-cuda-${CU_UPPERBOUND}.zip
            unzip tensorrt-rtx-${TRT_RTX_VERSION}.win10-amd64-cuda-${CU_UPPERBOUND}.zip
            rtx_lib_dir=${PWD}/TensorRT-RTX-${TRT_RTX_VERSION}/lib
            rtx_bin_dir=${PWD}/TensorRT-RTX-${TRT_RTX_VERSION}/bin
            export PATH=${rtx_lib_dir}:${rtx_bin_dir}:$PATH
            echo "PATH: $PATH"
            if [[ ${install_wheel_or_not} == true ]]; then
                pip install TensorRT-RTX-${TRT_RTX_VERSION}/python/tensorrt_rtx-${TRT_RTX_VERSION}-${CPYTHON_TAG}-none-win_amd64.whl
            fi
            # clean up the downloaded rtx zip
            rm tensorrt-rtx*.zip
        else
            curl -L https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.2/tensorrt-rtx-${TRT_RTX_VERSION}-linux-x86_64-cuda-${CU_UPPERBOUND}-release-external.tar.gz -o tensorrt-rtx-${TRT_RTX_VERSION}-linux-x86_64-cuda-${CU_UPPERBOUND}-release-external.tar.gz
            tar -xzf tensorrt-rtx-${TRT_RTX_VERSION}-linux-x86_64-cuda-${CU_UPPERBOUND}-release-external.tar.gz
            rtx_lib_dir=${PWD}/TensorRT-RTX-${TRT_RTX_VERSION}/lib
            rtx_bin_dir=${PWD}/TensorRT-RTX-${TRT_RTX_VERSION}/bin
            export LD_LIBRARY_PATH=${rtx_lib_dir}:${rtx_bin_dir}:$LD_LIBRARY_PATH
            echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
            if [[ ${install_wheel_or_not} == true ]]; then
                pip install TensorRT-RTX-${TRT_RTX_VERSION}/python/tensorrt_rtx-${TRT_RTX_VERSION}-${CPYTHON_TAG}-none-linux_x86_64.whl
            fi
            # clean up the downloaded rtx tarball
            rm tensorrt-rtx*.tar.gz
        fi
    else
        echo "It is the standard tensorrt build, skip install tensorrt-rtx"
    fi

}
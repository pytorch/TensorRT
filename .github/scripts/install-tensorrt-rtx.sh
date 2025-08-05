
install_tensorrt_rtx() {
    if [[ ${USE_RTX} == true ]]; then
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
            curl -L https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.0/TensorRT-RTX-1.0.0.21.Windows.win10.cuda-12.9.zip -o TensorRT-RTX-1.0.0.21.Windows.win10.cuda-12.9.zip
            unzip TensorRT-RTX-1.0.0.21.Windows.win10.cuda-12.9.zip
            rtx_lib_dir=${PWD}/TensorRT-RTX-1.0.0.21/lib
            export LD_LIBRARY_PATH=${rtx_lib_dir}:$LD_LIBRARY_PATH
            echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
            if [[ ${install_wheel_or_not} == true ]]; then
                pip install TensorRT-RTX-1.0.0.21/python/tensorrt_rtx-1.0.0.21-${CPYTHON_TAG}-none-win_amd64.whl
            fi
        else
            curl -L https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.0/TensorRT-RTX-1.0.0.21.Linux.x86_64-gnu.cuda-12.9.tar.gz -o TensorRT-RTX-1.0.0.21.Linux.x86_64-gnu.cuda-12.9.tar.gz
            tar -xzf TensorRT-RTX-1.0.0.21.Linux.x86_64-gnu.cuda-12.9.tar.gz
            rtx_lib_dir=${PWD}/TensorRT-RTX-1.0.0.21/lib
            export LD_LIBRARY_PATH=${rtx_lib_dir}:$LD_LIBRARY_PATH
            echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
            if [[ ${install_wheel_or_not} == true ]]; then
                pip install TensorRT-RTX-1.0.0.21/python/tensorrt_rtx-1.0.0.21-${CPYTHON_TAG}-none-linux_x86_64.whl
            fi
        fi
    else
        echo "It is the standard tensorrt build, skip install tensorrt-rtx"
    fi

}
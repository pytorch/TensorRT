export LD_LIBRARY_PATH=/opt/torch-tensorrt-builds/TensorRT-10.0.0.6/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
python -m pip install /opt/torch-tensorrt-builds/TensorRT-10.0.0.6/python/tensorrt-10.0.0b6-cp38-none-linux_x86_64.whl
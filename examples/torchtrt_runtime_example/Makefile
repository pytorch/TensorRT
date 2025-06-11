CXX=g++
DEP_DIR=$(PWD)/deps
CUDA_HOME=/usr/local/cuda
INCLUDE_DIRS=-I$(DEP_DIR)/libtorch/include -I$(DEP_DIR)/torch_tensorrt/include -I$(CUDA_HOME)/include -I$(DEP_DIR)/libtorch/include/torch/csrc/api/include
LIB_DIRS=-L$(DEP_DIR)/torch_tensorrt/lib -L$(DEP_DIR)/libtorch/lib -Wl,-rpath $(DEP_DIR)/tensorrt/lib
LIBS=-Wl,--no-as-needed -ltorchtrt_runtime -ltorchtrt_plugins  -Wl,--as-needed -ltorch -ltorch_cuda -ltorch_cpu -ltorch_global_deps -lbackend_with_compiler -lc10 -lc10_cuda
SRCS=main.cpp

TARGET=torchtrt_runtime_example

$(TARGET):
	$(CXX) $(SRCS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) -o $(TARGET)
	echo "Add to LD_LIBRARY_PATH: $(DEP_DIR)/torch_tensorrt/lib:$(DEP_DIR)/libtorch/lib:$(DEP_DIR)/tensorrt/lib:$(CUDA_HOME)/lib64"

generate_pt2:
	uv run network.py

clean:
	$(RM) $(TARGET)

CXX=g++
DEP_DIR=$(PWD)/deps
ROOT_DIR?="../../../" # path to TRTorch directory (including TRTorch)
INCLUDE_DIRS=-I$(DEP_DIR)/libtorch/include -I$(DEP_DIR)/libtorch/include/torch/csrc/api/include/ -I$(DEP_DIR)/TensorRT-8.0.1.6/include/ -I$(ROOT_DIR) -I /usr/local/cuda-11.1/include -I$(DEP_DIR)/trtorch/include
LIB_DIRS=-L$(DEP_DIR)/trtorch/lib -L$(DEP_DIR)/libtorch/lib -L /usr/local/cuda-11.1/lib64
LIBS=-Wl,--no-as-needed -ltrtorch -Wl,--as-needed -ltorch -ltorch_cuda -ltorch_cpu -ltorch_global_deps -lbackend_with_compiler -lc10 -lc10_cuda -lpthread -lcudart
SRCS=../datasets/cifar10.cpp ../benchmark/benchmark.cpp main.cpp

TARGET=qat

$(TARGET):
	$(CXX) $(SRCS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) -o $(TARGET)

clean:
	$(RM) $(TARGET)

CXX=g++
DEP_DIR=$(PWD)/deps
INCLUDE_DIRS=-I$(DEP_DIR)/libtorch/include -I$(DEP_DIR)/torch_tensorrt/include
LIB_DIRS=-L$(DEP_DIR)/torch_tensorrt/lib -L$(DEP_DIR)/libtorch/lib # -Wl,-rpath $(DEP_DIR)/tensorrt/lib -Wl,-rpath $(DEP_DIR)/cudnn/lib64
LIBS=-Wl,--no-as-needed -ltorchtrt_runtime -Wl,--as-needed -ltorch -ltorch_cuda -ltorch_cpu -ltorch_global_deps -lbackend_with_compiler -lc10 -lc10_cuda
SRCS=main.cpp

TARGET=torchtrt_runtime_example

$(TARGET):
	$(CXX) $(SRCS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) -o $(TARGET)

clean:
	$(RM) $(TARGET)

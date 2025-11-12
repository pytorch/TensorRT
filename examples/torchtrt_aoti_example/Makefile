CXX=g++
SITE_PACKAGES=$(shell python -c 'import site; print(site.getsitepackages()[0])')
CUDA_HOME=/usr/local/cuda-13.0

# gdb-friendly debug flags
CXXFLAGS?=-O0 -g3 -ggdb3 -fno-omit-frame-pointer -fno-optimize-sibling-calls
LDFLAGS?=-g3

INCLUDE_DIRS=-I$(SITE_PACKAGES)/torch/include -I$(SITE_PACKAGES)/torch_tensorrt/include -I$(CUDA_HOME)/include -I$(SITE_PACKAGES)/torch/include/torch/csrc/api/include

LIB_DIRS=-L$(SITE_PACKAGES)/torch_tensorrt/lib -L$(SITE_PACKAGES)/torch/lib -L/home/lanl/git/py312/TensorRT/py/torch_tensorrt/lib \
         -Wl,-rpath,$(SITE_PACKAGES)/tensorrt_libs -Wl,-rpath,$(SITE_PACKAGES)/torch/lib -Wl,-rpath,$(SITE_PACKAGES)/torch_tensorrt/lib -Wl,-rpath,$(CUDA_HOME)/lib64 -Wl,-rpath,/home/lanl/git/py312/TensorRT/py/torch_tensorrt/lib
LIBS=-Wl,--no-as-needed -ltorchtrt_runtime -ltorchtrt_plugins  -Wl,--as-needed -ltorch -ltorch_cuda -ltorch_cpu -ltorch_global_deps -ltorch_cuda_linalg -lc10 -lc10_cuda -lshm -ltorch_global_deps -ltorch_python

SRCS=inference.cpp

TARGET=torchtrt_aoti_example

$(TARGET): *cpp
	$(CXX) $(CXXFLAGS) $(SRCS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $(LDFLAGS) -o $(TARGET)
	echo "\n\nAdd to LD_LIBRARY_PATH: $(SITE_PACKAGES)/torch_tensorrt/lib:$(SITE_PACKAGES)/torch/lib:$(SITE_PACKAGES)/tensorrt_libs:$(CUDA_HOME)/lib64"

generate_pt2:
	python model.py

clean:
	$(RM) $(TARGET)

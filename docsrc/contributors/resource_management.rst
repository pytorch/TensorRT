.. _resource_management:

Resource Management
===================

Overview
--------

Efficient control of CPU and GPU memory is essential for successful model compilation, 
especially when working with large models such as LLMs or diffusion models. 
Uncontrolled memory growth can cause compilation failures or process termination. 
This guide describes the symptoms of excessive memory usage and provides methods 
to reduce both CPU and GPU memory consumption.

Memory Usage Control
--------------------

CPU Memory
^^^^^^^^^^

By default, Torch-TensorRT may consume up to **5×** the model size in CPU memory.  
This can exceed system limits when compiling large models.

**Common symptoms of high CPU memory usage:**

- Program freeze  
- Process terminated by the operating system  

**Ways to lower CPU memory usage:**

1. **Enable memory trimming**

   Set the following environment variable:

   .. code-block:: bash

      export TRIM_CPU_MEMORY=1

   This reduces approximately **2×** of redundant model copies, limiting 
   total CPU memory usage to up to **3×** the model size.

2. **Disable CPU offloading**

   In compilation settings, set:

   .. code-block:: python

      offload_module_to_cpu = False

   This removes another **1×** model copy, reducing peak CPU memory 
   usage to about **2×** the model size.

GPU Memory
^^^^^^^^^^

By default, Torch-TensorRT may consume up to **2×** the model size in GPU memory.

**Common symptoms of high GPU memory usage:**

- CUDA out-of-memory errors  
- TensorRT compilation errors  

**Ways to lower GPU memory usage:**

1. **Enable offloading to CPU**

   In compilation settings, set:

   .. code-block:: python

      offload_module_to_cpu = True

   This shifts one model copy from GPU to CPU memory.  
   As a result, peak GPU memory usage decreases to about **1×** 
   the model size, while CPU memory usage increases by roughly **1×**.



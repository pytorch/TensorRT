.. _dev_infra:

PyTorch CI
====================

Our main CI provider is the PyTorch CI, backed by `pytorch/test-infra <https://github.com/pytorch/test-infra>`_


Debugging CI Failures
------------------------

Sometimes, you may observe errors on CI tests but cannot repro on a local machines. There are a few possible reasons:

- Oversubscription of resources issue that means CI runs too many jobs at the same time. You have to reduce the num of parallel jobs by lowering -n like ``python -m pytest -ra --junitxml=${RUNNER_TEST_RESULTS_DIR}/dynamo_converters_test_results.xml -n 8 conversion/``
- Your ENV may be different from CI.
    - GPU arch. As of 11/20/2024, our CI is using AWS linux.g5.4xlarge.nvidia.gpu which runs on 1 x A10 GPU
    - Torch-TensorRT version
    - Dependency versions (PyTorch, TensorRT, etc.)


Create a same environment as CI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CI builds are slightly different than local builds since they will use PyTorch versions that are likely newer than the ones you have installed locally.
Therefore when debugging, it may be helpful to replicate the CI environment.

We build all CI wheels using a AlmaLinux manylinux container customized by PyTorch: ``pytorch/manylinux2_28-builder:cudaXX.X`` e.g.,``pytorch/manylinux2_28-builder:cuda12.8``
This container is available on Docker Hub and can be pulled using the following command:

.. code-block:: bash

    docker pull pytorch/manylinux2_28-builder:cuda12.4

You can then either download builds from CI for testing:

.. image:: /contributors/images/ci_whls.png
   :width: 512px
   :height: 512px
   :scale: 50 %
   :align: right


.. code-block:: bash


    /opt/python/cp311-cp311m/bin/python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
    /opt/python/cp311-cp311m/bin/python -m pip install docker_workspace/torch_tensorrt-2.6.0.dev20241119+cu124-cp311-cp311m-linux_x86_64.whl    # Install your downloaded artifact
    # enter Torch-TRT dir and run: pip install -r requirements-dev.txt
    /opt/python/cp311-cp311m/bin/python -m pip install timm pytest-xdist   # pytest-xdist is used by pytest to parallel tasks



Or you can replicate the build in container by running the following command

.. code-block:: bash

    docker run --rm -it -v $(pwd):/workspace pytorch/manylinux2_28-builder:cuda12.8 bash
    # Inside container
    cd /workspace
    export CUDA_HOME=/usr/local/cuda-12.8
    export CI_BUILD=1
    ./packaging/pre_build_script.sh
    /opt/python/cp311-cp311m/bin/python setup.py bdist_wheel
    /opt/python/cp311-cp311m/bin/python -m pip install timm pytest-xdist   # pytest-xdist is used by pytest to parallel tasks



Run CI Tests in the same manner as PyTorch CI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash
    export RUNNER_TEST_RESULTS_DIR=/tmp/test_results

    export USE_HOST_DEPS=1
    export CI_BUILD=1
    cd tests/py/dynamo
    python -m pytest -ra --junitxml=${RUNNER_TEST_RESULTS_DIR}/dynamo_converters_test_results.xml -n 4 conversion/

Building Torch-TensorRT as Hermetically As Possible
---------------------------------------------------

Torch-TensorRT uses a combination of `Bazel <https://bazel.build/>`_ and `UV <https://docs.astral.sh/uv>`_ to build the project in a (near) hermetic manner.

C++ Dependencies are declared in ``MODULE.bzl`` using ``http_archive`` and `` git_repository`` rules. Using a combination of ``pyproject.toml`` and ``uv``
we lock python dependencies as well. This insures that the dependencies fetched will be identical on each build. Using the build command
``uv pip install -e . `` or ``uv run <script using torch_tensorrt>`` will use these dependencies to build the project. When providing a reproducer for a
locally identified bug, providing the `MODULE.bzl` and `pyproject.toml` files will help us reproduce the issue.

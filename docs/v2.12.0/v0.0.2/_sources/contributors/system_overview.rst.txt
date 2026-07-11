.. _system_overview:

System Overview
================

TRTorch is primarily a C++ Library with a Python API planned. We use Bazel as our build system and target Linux x86_64 and
Linux aarch64 (only natively) right now. The compiler we use is GCC 7.5.0 and the library is untested with compilers before that
version so there may be compilation errors if you try to use an older compiler.

The repository is structured into:

* core: Main compiler source code
* cpp: C++ API
* tests: tests of the C++ API, the core and converters
* docs: Documentation
* docsrc: Documentation Source
* third_party: BUILD files for dependency libraries

The C++ API is unstable and subject to change until the library matures, though most work is done under the hood in the core.

The core has a couple major parts: The top level compiler interface which coordinates ingesting a module, lowering,
converting and generating a new module and returning it back to the user. The there are the three main phases of the
compiler, the lowering phase, the conversion phase, and the execution phase.

.. include:: phases.rst
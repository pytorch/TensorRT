.. _partitioning:

Partitioning Phase
====================

The phase is optional and enabled by the user. It instructs the compiler to seperate nodes into ones that should run in PyTorch and ones that should run in TensorRT.
Criteria for seperation include: Lack of a converter, operator is explicitly set to run in PyTorch by the user or the node has a flag which tells partitioning to
run in PyTorch by the module fallback passes.
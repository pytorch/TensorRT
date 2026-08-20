.. _writing_dynamo_aten_lowering_passes:

Writing Dynamo ATen Lowering Passes
===================

Basics of a Lowering Pass
------------

ATen lowering passes are Python functions which take as input a graph of ATen operators, apply some desired modification such as operator coalescing/fusion, operator replacement, subgraph rewriting, custom operator insertion, or other operation on a `torch.fx.GraphModule`, then return the modified graph to the caller. These lowering passes generally modify the graph in-place and return the same input object.

Lowering Pass Requirements
------------

An ATen lowering pass function in Torch-TRT must satisfy two requirements:
- The function must take as input a `torch.fx.GraphModule` and a sequence of torch Tensors, `Sequence[torch.Tensor]`, and return the lowered `torch.fx.GraphModule`
- The function must leave the graph in a valid and invoke-able state, including performing any necessary linting and recompilation

See this link for information on `Graph Manipulations <https://pytorch.org/docs/stable/fx.html#graph-manipulation>`_ in FX. See below for an example of a lowering pass which repairs graphs that have inputs which are also outputs, a disallowed configuration for TRT Engines.

Example Lowering Pass
------------

.. code-block:: python

    def repair_input_as_output(gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]) -> torch.fx.GraphModule:
        """Repair scenarios where inputs are also outputs of the graph

        TRT does not allow such cases, so we insert a clone (identity) layer
        """
        modified_graph = False

        # Extract graph placeholder Tensors
        placeholders = [
            node
            for node in gm.graph.nodes
            if (
                node.op == "placeholder"
                and isinstance(node.type, type)
                and issubclass(node.type, torch.Tensor)
            )
        ]

        for placeholder in placeholders:
            # If any placeholder has any users which are direct graph outputs
            if len(placeholder.users) >= 1 and any(
                user.op == "output" for user in placeholder.users
            ):
                modified_graph = True

                # Get direct graph outputs which are direct uses of placeholders
                direct_outputs = [user for user in placeholder.users if user.op == "output"]

                # Insert clone node for placeholder to ensure
                # placeholder is not a direct output
                with gm.graph.inserting_after(placeholder):
                    cloned_placeholder = gm.graph.call_function(
                        torch.ops.aten.clone.default,
                        args=(placeholder,),
                    )

                # Replace placeholder as output with cloned version
                for output in direct_outputs:
                    output.replace_input_with(placeholder, cloned_placeholder)

        # If the graph was modified, clean up the graph and ensure it is up-to-date
        if modified_graph:
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()
            logger.debug(f"Graph after repair_input_as_output:\n{gm.graph}")

        return gm


Registering Lowering Passes
----------------------

Lowering passes are currently registered in `py/torch_tensorrt/dynamo/lowering/passes/__init__.py`, using the `torch.fx.passes.pass_manager.PassManager` utility to assemble the list of passes in a desired order. New passes added directly to that list will be applied to graphs in the Torch-TensorRT `torch.compile` backend. Currently, we offer an ATen lowering pass registration decorator for convenience, which can be invoked either directly, or with the optional `index` keyword argument which controls where in the pass list the lowering pass will be inserted.

For instance, to insert the pass at the default location (end of the list), the following code can be used:

.. code-block:: python

    @_aten_lowering_pass
    def my_custom_pass(gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]) -> torch.fx.GraphModule:
        ...

Alternatively, to insert the pass at a custom index (such as the front of the list) in the passlist, the following code can be used:

.. code-block:: python

    @_aten_lowering_pass(index=0)
    def my_custom_pass(gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]) -> torch.fx.GraphModule:
        ...

There are also provided utilities in `torch_tensorrt.dynamo.lowering.passes` for displaying the currently-available lowering pass list, applying those passes to an arbitrary `torch.fx.GraphModule`, and removing the lowering pass at a specific index.

.. code-block:: python

    # Print all lowering passes in the list
    print(dump_lowering_passes())

    # Apply lowering passes to a GraphModule
    apply_lowering_passes(graph_module, sample_inputs)

    # Remove the lowering pass at index 1
    _remove_lowering_pass(index=1)

**Note:** The above APIs are subject to change, as the lowering pass system evolves.

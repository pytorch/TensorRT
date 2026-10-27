# Numerical and boundary correctness

The captured PyTorch graph defines the computation. Read operation arguments,
broadcasting, reduction axes, scalar constants, shapes, and output ordering.
Preserve intermediate dtype conversions and rounding; algebraic equivalence
over real numbers does not guarantee equivalence for floating-point tensors.

For float16 and bfloat16 reference operations, restore the reference result's
dtype before consuming it in a later operation. Float32 fusion or reassociation
can also change results. Apply the actual configured tolerances; never change
the reference or invent more permissive tolerances.

Use the backend's tensor ABI and parameter order. Account for all elements and
all outputs, including partial blocks and broadcasted dimensions. Avoid reading
or writing beyond tensor storage. Inputs are immutable unless the supplied
contract explicitly says otherwise. Output allocation and host launch wrappers
belong to the executor under the built-in backends.

Respect the selected nonfinite policy: NaNs, infinities, signed values, and
reductions can expose errors hidden by ordinary random inputs. Diagnose a
failure from the evaluator's case, dtype, output, and error details. A few
passing cases are observed checks, not a proof over all inputs.

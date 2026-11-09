# Numerical and boundary correctness

The captured PyTorch graph defines the computation. Read operation arguments,
broadcasting, reduction axes, scalar constants, shapes, and output ordering.
Preserve intermediate dtype conversions and rounding; algebraic equivalence
over real numbers does not guarantee equivalence for floating-point tensors.

For float16 and bfloat16 reference operations, restore the reference result's
dtype before consuming it in a later operation. Float32 fusion or reassociation
can also change results. Apply the actual configured tolerances; never change
the reference or invent more permissive tolerances.

Without configured tolerances, outputs are compared with the exact result: the
reference evaluated in float64, with every floating dtype in the graph,
including its own float16, bfloat16 and float32 casts, widened to float64. It
runs on the GPU, or on the CPU when the GPU lacks memory or a float64 kernel; a
region whose inputs and outputs are all float16 or bfloat16 may use the same
promotion to float32 instead. The exact result never uses TF32, whatever the
process's matmul settings are. Eager PyTorch is measured against the same
exact result, also without TF32 for matmuls and convolutions, and the
candidate may err as much as eager does, not more:

- Element by element: within `atol + rtol*|exact|` (the defaults of
  `torch.testing.assert_close` for the output dtype) plus 4 times eager's own
  error, which is a few units in the last place where eager only rounds and
  eager's largest error over the output where it accumulates.
- Over the whole output: the error's L2 norm at most 2 times eager's, or 2
  times a floor, whichever is larger: one unit roundoff of the exact values
  (never less than about 1e-6 relative), and never less than `atol` per
  element. A wrong scale, decay or window shifts many elements a little; this
  bound catches it even when every element is within its own allowance.

Match eager accuracy, not necessarily its exact bits. Keep reductions and
accumulations in float32, as eager does; a candidate that is markedly less
accurate than eager fails even when it is approximately right.

Float32 matrix multiplication and convolution must be IEEE accurate unless the
task's validation sets `fp32_matmul_precision` to `"high"`: then TF32 is
allowed for both, and a candidate may err as much as the reference does when
PyTorch runs it with TF32 matmuls and convolutions.

With configured `rtol` and `atol`, outputs are compared with eager PyTorch's
output at those tolerances instead.

Use the backend's tensor ABI and parameter order. Account for all elements and
all outputs, including partial blocks and broadcasted dimensions. Avoid reading
or writing beyond tensor storage. Validation surrounds tensors with guard
bands, so an out-of-bounds write fails the correctness stage; an unmasked read
fails only when it changes an output, but is still invalid. Unless the task's
`codegen.aligned_io` is true, a tensor's address is a multiple of its element
size only. Inputs are
immutable unless the supplied contract explicitly says otherwise. Output
allocation and host launch wrappers belong to the executor under the built-in
backends.

Validation inputs come from the region's captured samples, its real inputs
from the user's model, when the task has them. Every synthetic case starts
from a sample:

- Random cases scale each continuous value by a random factor and keep it
  within the sampled range of its input, so signs and zeros hold. The first
  random case keeps all remaining inputs exactly as the sample has them; the
  later ones vary those inputs too. Integer-valued inputs, such as positions
  and window bounds, become random integers within the sampled range, so a
  window can end anywhere, not only on a tile boundary. 0/1 masks become
  random masks with the sample's share of ones. Both are sorted when every
  sample's elements never decrease when flattened, so window bounds keep
  start <= end and left padding stays contiguous. Inputs that hold a single
  value, such as a weight of ones or a scalar setting, and inputs of fewer
  than 8 elements get random positive factors; their zeros become random
  values of the input's sign.
- Magnitude cases multiply every input that is not integer-valued by 1e-3 and
  by 1e2, and offset it by +100 or -100. They never change an input's sign:
  an input whose samples are never negative is only offset upwards, one whose
  samples are never positive only downwards.
- `scaled x100`, `offset +100` and `offset -100` take inputs beyond the
  sampled range, where a correct kernel's standard arithmetic, such as
  exponentials of differences of long prefix sums, can lose far more accuracy
  than eager PyTorch. They are robustness cases:
  - wherever eager PyTorch's output is finite, the candidate's must be finite
    too, which an unguarded exponential or an overflowing intermediate breaks;
  - the error over an output may reach 2 times eager's or 1% of the exact
    output's L2 norm, whichever is larger;
  - each element may err by 4 times eager's largest error or 10% of the
    output's largest exact magnitude, whichever is larger, plus `atol`, so a
    few grossly wrong elements cannot hide in the norm of a large output.

  Every other case, `scaled x0.001` included, uses the bounds above. With
  configured `rtol` and `atol`, every case uses those tolerances instead.
- Constant cases set each input to 0, -1 or 1 where its sign allows.

Read every input; do not hard-code a value that the example happens to hold,
and do not assume that window bounds, masks or scalar inputs keep their
example values. Handle magnitudes beyond the samples' without overflowing
(for example, subtract the row maximum before exponentiating). Without
samples, inputs are standard normal and every case, the magnitude cases
included, uses the in-range bounds. When eager PyTorch's outputs are then NaN
or infinite at more than half of an output's elements on every random case,
such inputs leave the region's domain, and the evaluation stops as a `setup`
failure: compile with real example tensors or pass `validation_inputs`.

Where eager PyTorch's output is NaN or infinite, any candidate value passes;
everywhere else the candidate must be finite and accurate. When the nonfinite
policy injects a NaN or an infinity into an input, outputs that the injected
value reaches may also be NaN or infinite, but a finite value there must be
accurate, and outputs it does not reach must be unaffected. Every input must
be read: where eager PyTorch's outputs turn NaN or infinite at elements that
depend on an injected NaN (they are finite with the element's original value
and change when it changes), the candidate must be NaN or infinite at one of
them at least, or, if its arithmetic drops the NaN, as a maximum or clamp
with `fmaxf` or `tl.maximum` semantics does, its outputs there must change
when the element takes another finite value. A kernel that ignores an input
or hard-codes a value the example happens to hold fails this check. A masked
kernel need not read an element that eager PyTorch multiplies by zero, and is
not held to it there. Diagnose a failure from the evaluator's case, dtype,
output, and error details. A few passing cases are observed checks, not a
proof over all inputs.

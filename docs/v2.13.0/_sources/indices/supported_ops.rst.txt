
.. _supported_ops:

=================================
Operators Supported
=================================

.. note::

   This page reflects operator coverage for the **Dynamo** (``torch_tensorrt.dynamo``) frontend.
   Operators marked *Converted* have a native Dynamo converter.
   Operators marked *Lowered* are handled via ATen decompositions before reaching TensorRT.

ATen Core Ops — Converted
-------------------------

*156 operators with native Dynamo converters.*

- aten._adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor
- aten._adaptive_avg_pool3d(Tensor self, SymInt[3] output_size) -> Tensor
- aten._cdist_forward(Tensor x1, Tensor x2, float p, int? compute_mode) -> Tensor
- aten._embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
- aten._native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
- aten._native_batch_norm_legit_no_training(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)
- aten._pdist_forward(Tensor self, float p=2) -> Tensor
- aten._softmax(Tensor self, int dim, bool half_to_float) -> Tensor
- aten._to_copy(Tensor self, \*, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
- aten.abs(Tensor self) -> Tensor
- aten.acos(Tensor self) -> Tensor
- aten.acosh(Tensor self) -> Tensor
- aten.add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
- aten.add.Tensor(Tensor self, Tensor other, \*, Scalar alpha=1) -> Tensor
- aten.addmm(Tensor self, Tensor mat1, Tensor mat2, \*, Scalar beta=1, Scalar alpha=1) -> Tensor
- aten.amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
- aten.amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
- aten.any(Tensor self) -> Tensor
- aten.any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
- aten.any.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
- aten.arange.start_step(Scalar start, Scalar end, Scalar step=1, \*, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
- aten.argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
- aten.argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
- aten.as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)
- aten.asin(Tensor self) -> Tensor
- aten.asinh(Tensor self) -> Tensor
- aten.atan(Tensor self) -> Tensor
- aten.atan2(Tensor self, Tensor other) -> Tensor
- aten.atan2.out(Tensor self, Tensor other, \*, Tensor(a!) out) -> Tensor(a!)
- aten.atanh(Tensor self) -> Tensor
- aten.avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
- aten.avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
- aten.bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor
- aten.bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor
- aten.bitwise_not(Tensor self) -> Tensor
- aten.bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor
- aten.bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor
- aten.bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor
- aten.bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor
- aten.bmm(Tensor self, Tensor mat2) -> Tensor
- aten.cat(Tensor[] tensors, int dim=0) -> Tensor
- aten.ceil(Tensor self) -> Tensor
- aten.clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
- aten.clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
- aten.clone(Tensor self, \*, MemoryFormat? memory_format=None) -> Tensor
- aten.constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor
- aten.convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor
- aten.copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor
- aten.cos(Tensor self) -> Tensor
- aten.cosh(Tensor self) -> Tensor
- aten.cumsum(Tensor self, int dim, \*, ScalarType? dtype=None) -> Tensor
- aten.diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
- aten.div.Scalar(Tensor self, Scalar other) -> Tensor
- aten.div.Scalar_mode(Tensor self, Scalar other, \*, str? rounding_mode) -> Tensor
- aten.div.Tensor(Tensor self, Tensor other) -> Tensor
- aten.div.Tensor_mode(Tensor self, Tensor other, \*, str? rounding_mode) -> Tensor
- aten.elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
- aten.embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
- aten.empty.memory_format(SymInt[] size, \*, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
- aten.eq.Scalar(Tensor self, Scalar other) -> Tensor
- aten.eq.Tensor(Tensor self, Tensor other) -> Tensor
- aten.erf(Tensor self) -> Tensor
- aten.exp(Tensor self) -> Tensor
- aten.expand(Tensor(a) self, SymInt[] size, \*, bool implicit=False) -> Tensor(a)
- aten.expm1(Tensor self) -> Tensor
- aten.flip(Tensor self, int[] dims) -> Tensor
- aten.floor(Tensor self) -> Tensor
- aten.fmod.Scalar(Tensor self, Scalar other) -> Tensor
- aten.fmod.Tensor(Tensor self, Tensor other) -> Tensor
- aten.full(SymInt[] size, Scalar fill_value, \*, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
- aten.gather(Tensor self, int dim, Tensor index, \*, bool sparse_grad=False) -> Tensor
- aten.ge.Scalar(Tensor self, Scalar other) -> Tensor
- aten.ge.Tensor(Tensor self, Tensor other) -> Tensor
- aten.gelu(Tensor self, \*, str approximate='none') -> Tensor
- aten.grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
- aten.gt.Scalar(Tensor self, Scalar other) -> Tensor
- aten.gt.Tensor(Tensor self, Tensor other) -> Tensor
- aten.index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
- aten.index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
- aten.index_select(Tensor self, int dim, Tensor index) -> Tensor
- aten.isinf(Tensor self) -> Tensor
- aten.isnan(Tensor self) -> Tensor
- aten.le.Scalar(Tensor self, Scalar other) -> Tensor
- aten.le.Tensor(Tensor self, Tensor other) -> Tensor
- aten.leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
- aten.log(Tensor self) -> Tensor
- aten.log10(Tensor self) -> Tensor
- aten.log1p(Tensor self) -> Tensor
- aten.log2(Tensor self) -> Tensor
- aten.logical_and(Tensor self, Tensor other) -> Tensor
- aten.logical_not(Tensor self) -> Tensor
- aten.logical_or(Tensor self, Tensor other) -> Tensor
- aten.logical_xor(Tensor self, Tensor other) -> Tensor
- aten.lt.Scalar(Tensor self, Scalar other) -> Tensor
- aten.lt.Tensor(Tensor self, Tensor other) -> Tensor
- aten.max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
- aten.maximum(Tensor self, Tensor other) -> Tensor
- aten.mean(Tensor self, \*, ScalarType? dtype=None) -> Tensor
- aten.mean.dim(Tensor self, int[1]? dim, bool keepdim=False, \*, ScalarType? dtype=None) -> Tensor
- aten.min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
- aten.minimum(Tensor self, Tensor other) -> Tensor
- aten.mm(Tensor self, Tensor mat2) -> Tensor
- aten.mul.Scalar(Tensor self, Scalar other) -> Tensor
- aten.mul.Tensor(Tensor self, Tensor other) -> Tensor
- aten.native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)
- aten.native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)
- aten.native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
- aten.ne.Scalar(Tensor self, Scalar other) -> Tensor
- aten.ne.Tensor(Tensor self, Tensor other) -> Tensor
- aten.neg(Tensor self) -> Tensor
- aten.nonzero(Tensor self) -> Tensor
- aten.permute(Tensor(a) self, int[] dims) -> Tensor(a)
- aten.pow.Scalar(Scalar self, Tensor exponent) -> Tensor
- aten.pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
- aten.pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
- aten.prod(Tensor self, \*, ScalarType? dtype=None) -> Tensor
- aten.prod.dim_int(Tensor self, int dim, bool keepdim=False, \*, ScalarType? dtype=None) -> Tensor
- aten.rand(SymInt[] size, \*, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
- aten.randn(SymInt[] size, \*, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
- aten.randperm(SymInt n, \*, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
- aten.reflection_pad1d(Tensor self, SymInt[2] padding) -> Tensor
- aten.reflection_pad2d(Tensor self, SymInt[4] padding) -> Tensor
- aten.reflection_pad3d(Tensor self, SymInt[6] padding) -> Tensor
- aten.relu(Tensor self) -> Tensor
- aten.remainder.Scalar(Tensor self, Scalar other) -> Tensor
- aten.remainder.Tensor(Tensor self, Tensor other) -> Tensor
- aten.replication_pad2d(Tensor self, SymInt[4] padding) -> Tensor
- aten.replication_pad3d(Tensor self, SymInt[6] padding) -> Tensor
- aten.resize_(Tensor(a!) self, SymInt[] size, \*, MemoryFormat? memory_format=None) -> Tensor(a!)
- aten.round(Tensor self) -> Tensor
- aten.scalar_tensor(Scalar s, \*, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
- aten.scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
- aten.scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
- aten.select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)
- aten.sigmoid(Tensor self) -> Tensor
- aten.sign(Tensor self) -> Tensor
- aten.sin(Tensor self) -> Tensor
- aten.sinh(Tensor self) -> Tensor
- aten.slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
- aten.sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
- aten.split_with_sizes(Tensor(a -> \*) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]
- aten.sqrt(Tensor self) -> Tensor
- aten.squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
- aten.squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)
- aten.sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
- aten.sub.Tensor(Tensor self, Tensor other, \*, Scalar alpha=1) -> Tensor
- aten.sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, \*, ScalarType? dtype=None) -> Tensor
- aten.sym_numel(Tensor self) -> SymInt
- aten.tan(Tensor self) -> Tensor
- aten.tanh(Tensor self) -> Tensor
- aten.topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
- aten.trunc(Tensor self) -> Tensor
- aten.unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
- aten.upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
- aten.upsample_nearest2d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
- aten.where.self(Tensor condition, Tensor self, Tensor other) -> Tensor

ATen Core Ops — Lowered via Decomposition
-----------------------------------------

*26 operators decomposed into supported primitives before TensorRT compilation.*

- aten._log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
- aten.adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor
- aten.alias(Tensor(a) self) -> Tensor(a)
- aten.avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor
- aten.col2im(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
- aten.embedding_dense_backward(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq) -> Tensor
- aten.empty_strided(SymInt[] size, SymInt[] stride, \*, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
- aten.fill.Scalar(Tensor self, Scalar value) -> Tensor
- aten.full_like(Tensor self, Scalar fill_value, \*, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
- aten.hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
- aten.masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
- aten.native_group_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, SymInt N, SymInt C, SymInt HxW, int group, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
- aten.native_layer_norm_backward(Tensor grad_out, Tensor input, SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
- aten.reciprocal(Tensor self) -> Tensor
- aten.repeat(Tensor self, SymInt[] repeats) -> Tensor
- aten.rsqrt(Tensor self) -> Tensor
- aten.scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
- aten.scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, \*, bool include_self=True) -> Tensor
- aten.select_scatter(Tensor self, Tensor src, int dim, SymInt index) -> Tensor
- aten.slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor
- aten.sym_is_contiguous(Tensor self, MemoryFormat memory_format=contiguous_format) -> SymBool
- aten.sym_size.int(Tensor self, int dim) -> SymInt
- aten.sym_stride.int(Tensor self, int dim) -> SymInt
- aten.var.correction(Tensor self, int[1]? dim=None, \*, Scalar? correction=None, bool keepdim=False) -> Tensor
- aten.var.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
- aten.view(Tensor(a) self, SymInt[] size) -> Tensor(a)

Python Builtin Ops — Converted
------------------------------

*9 Python-level operators supported.*

- _operator.add
- _operator.eq
- _operator.floordiv
- _operator.getitem
- _operator.mod
- _operator.mul
- _operator.pow
- _operator.sub
- _operator.truediv

Prims Ops — Converted
---------------------

*3 prims operators with native Dynamo converters.*

- prims.broadcast_in_dim(Tensor(a) a, SymInt[] shape, int[] broadcast_dimensions) -> Tensor(a)
- prims.div(Tensor self, Tensor other) -> Tensor
- prims.sum(Tensor inp, int[]? dims, \*, ScalarType? output_dtype=None) -> Tensor

Prims Ops — Lowered via Decomposition
-------------------------------------

*1 prims operators decomposed into supported primitives.*

- prims.var(Tensor inp, int[]? dims, float? correction=1, \*, ScalarType? output_dtype=None) -> Tensor


import triton
import triton.language as tl

from typing import Tuple
import torch_tensorrt
from torch._subclasses.fake_tensor import FakeTensorMode


@triton.jit
def elementwise_mul_kernel(X, Y, Z, BLOCK_SIZE: tl.constexpr):
    # Program ID determines the block of data each thread will process
    pid = tl.program_id(0)
    # Compute the range of elements that this thread block will work on
    block_start = pid * BLOCK_SIZE
    # Range of indices this thread will handle
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Load elements from the X and Y tensors
    x_vals = tl.load(X + offsets)
    y_vals = tl.load(Y + offsets)
    # Perform the element-wise multiplication
    z_vals = x_vals * y_vals
    # Store the result in Z
    tl.store(Z + offsets, z_vals)


import torch
from torch.library import custom_op

#@torch_tensorrt.dynamo.conversion.plugin.custom_op("torchtrt_ex::elementwise_mul", mutates_args=())
@torch.library.custom_op("torchtrt_ex::elementwise_mul", mutates_args=())  # type: ignore[misc]
def elementwise_mul(X: torch.Tensor, Y: torch.Tensor, b: float=.2, a: int=2) -> torch.Tensor:
    # Ensure the tensors are on the GPU
    assert X.is_cuda and Y.is_cuda, "Tensors must be on CUDA device."
    assert X.shape == Y.shape, "Tensors must have the same shape."

    # Create output tensor
    Z = torch.empty_like(X)

    # Define block size
    BLOCK_SIZE = 1024

    # Grid of programs
    grid = lambda meta: (X.numel() // meta['BLOCK_SIZE'],)

    # Launch the kernel
    elementwise_mul_kernel[grid](X, Y, Z, BLOCK_SIZE=BLOCK_SIZE)

    return Z

from torch import nn


class MyModel(nn.Module):  # type: ignore[misc]
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.add(x, y)
        res = torch.ops.torchtrt_ex.elementwise_mul.default(x, z, a=1)

        return res


my_model = MyModel().to("cuda")
m = torch.full((64, 64), 2, device='cuda', dtype=torch.float)
n = torch.full((64, 64), 3, device='cuda', dtype=torch.float)

def mksym(shape_env, value, source, dynamic_dim):
    return shape_env.create_symintnode(
        shape_env.create_symbol(
            value,
            source=source,
            dynamic_dim=dynamic_dim,
        ),
        hint=value,
        source=source,
    )

@torch.library.register_fake("torchtrt_ex::elementwise_mul")
def _(x: torch.Tensor, y: torch.Tensor, b: float=.2, a: int=2) -> torch.Tensor:
    return x

import tensorrt_bindings.plugin as trtp
from torch._dynamo.source import LocalSource
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from sympy import lambdify


def generate_plugin(plugin_name : str):
    namespace, name = plugin_name.split("::")
    
    torch_op = getattr(getattr(torch.ops, namespace), name) # torch.ops.torchtrt_ex.elementwise_mul
    print(torch_op)
    # retrieve torch.ops.torchtrt_ex.elementwise_mul
    
    print(torch_op._schemas)
    
    # import pdb; pdb.set_trace();
    
    # def parse_torch_op_schema(torch_op):
    #     schema = torch_op._schemas['']
    #     args = []
    #     kwargs = {}
    #     for arg in schema.arguments:
    #         print(f"Name: {arg.name}, Type: {arg.type}, Default: {arg.default_value}")
            
    #     for ret in schema.returns:
    #         print(f"Return Type: {ret.type}")
    #         # if arg.default_value is None:
    #         #     args.append(arg.name)
    #         # else:
    #         #     kwargs[arg.name] = arg.default_value
    #     return args, kwargs
        
    # parse_torch_op_schema(torch_op)
    
    def _generic_plugin_desc_creator(torch_op):
        schema = torch_op._schemas['']
        
        tensor_args = []
        
        arg_list = []
        
        func_body = []
        
        func_body.append("  shape_env = ShapeEnv()")
        func_body.append("fake_mode = FakeTensorMode(shape_env=shape_env)")
        
        for arg in schema.arguments:
            print(arg.type)
            # import pdb; pdb.set_trace();
            arg_type = "trtp.TensorDesc" if arg.type.isSubtypeOf(torch._C.TensorType.get()) else arg.type
            arg_list.append(f"{arg.name} : {arg_type}")
            
            if arg.type.isSubtypeOf(torch._C.TensorType.get()):
                tensor_args.append(arg)
            
        for arg in tensor_args:
            func_body.append(f"sample_{arg.name} = {{f'{arg.name}{{i}}': 5 for i in range({arg.name}.ndim)}}")
            
        
        for arg in tensor_args:
            func_body.append(f"sysm_{arg.name} = [mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC) for k, v in sample_{arg.name}.items()]")
            
        func_body.append("with FakeTensorMode() as fake_mode:")
        
        for arg in tensor_args: 
            func_body.append(f"  fake_{arg.name} = torch.randn(sysm_{arg.name})")
            
        #     sample_x = {f"x{i}": 5 for i in range(x.ndim)}
        # sample_y = {f"y{i}": 5 for i in range(y.ndim)}
        # syms_x = [mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC) for k,v in sample_x.items()]
        # syms_y = [mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC) for k,v in sample_y.items()]
        # running_line = f"output = {torch_op}("
        running_args = []
        for arg in schema.arguments: 
            if arg in tensor_args:
                running_args.append(f"fake_{arg.name}")
            else:
                running_args.append(f"{arg.name}")
        running_line_args = ", ".join(running_args)
        running_line = f"  output = torch.ops.{torch_op}({running_line_args})"
        func_body.append(running_line)
            

        # Join the argument list to create the signature
        input_signature = ", ".join(arg_list)
        print(input_signature)
        
        ret_list = []
        for ret in schema.returns:
            print(ret.type)
            if ret.type.isSubtypeOf(torch._C.TensorType.get()):
                ret_list.append(f"trtp.TensorDesc")
            else: 
                raise Exception("Return type has be to Tensor for TRT plugin")
         
        
        ret_signature = "trtp.TensorDesc" if len(ret_list)  == 1 else f"Tuple[{', '.join(ret_list)}" 

        plugin_signature = f"def add_plugin_desc({input_signature}) -> {ret_signature}:"
        print(plugin_signature)
        
        
        body_str = "\n  ".join(func_body)
        print("-----------------\n")
        print(plugin_signature)
        print(body_str)
        print("\n-----------------\n")

    def generate_signature(torch_op):
        schema = torch_op._schemas['']
        tensor_args = []
        arg_list = []
        func_body = []
        
        func_body.append("  shape_env = ShapeEnv()")
        func_body.append("fake_mode = FakeTensorMode(shape_env=shape_env)")
        
        args = []
        kwargs = []
        
        for arg in schema.arguments:
            # import pdb; pdb.set_trace();
            arg_type = "trtp.TensorDesc" if arg.type.isSubtypeOf(torch._C.TensorType.get()) else arg.type
            # arg_list.append(f"{arg.name} : {arg_type}")
            arg_list.append(arg.name)
            
            if arg.type.isSubtypeOf(torch._C.TensorType.get()):
                tensor_args.append(arg)
                
            if arg.default_value is None:
                args.append(arg.name)
            else:
                kwargs.append(f"{arg.name} = {arg.default_value}")
                
        input_signature = ", ".join(arg_list)

        ret_list = []
        for ret in schema.returns:
            print(ret.type)
            if ret.type.isSubtypeOf(torch._C.TensorType.get()):
                ret_list.append(f"trtp.TensorDesc")
            else: 
                raise Exception("Return type has be to Tensor for TRT plugin")
         
        
        ret_signature = "trtp.TensorDesc" if len(ret_list)  == 1 else f"Tuple[{', '.join(ret_list)}" 

        plugin_signature = f"def add_plugin_desc({input_signature}):"
        args_input = ", ".join(args)
        kwargs_input = ", ".join(kwargs)
        # print(args_input)
        # print(kwargs_input)
        # print(plugin_signature)
        plugin_impl_arg_list = arg_list
        plugin_impl_arg_list.append('outputs')
        plugin_impl_arg_list.append('stream')
        plugin_impl_input = ", ".join(plugin_impl_arg_list)
        plugin_impl_signagture = f"def add_plugin_impl({plugin_impl_input}):"
        
        print(plugin_impl_signagture)
        
        return args_input, kwargs_input, plugin_signature, plugin_impl_signagture
        
    
    # _generic_plugin_desc_creator(torch_op)
    
    args_input, kwargs_input, plugin_signature, plugin_impl_signagture = generate_signature(torch_op)
    
    def _generic_plugin_desc(*args, **kwargs) -> trtp.TensorDesc:
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from sympy import lambdify
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        syms_args = []
        for arg in args:
            sample = {f"{i}": 5 for i in range(arg.ndim)}
            syms_arg = [mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC) for k,v in sample.items()]
            syms_args.append(syms_arg)
            
        with FakeTensorMode() as fake_mode:
            fake_args = []
            for syms_arg in syms_args:
                fake_arg = torch.randn(syms_arg)
                fake_args.append(fake_arg)
                
            output = torch_op(*fake_args, **kwargs)
                
        # We assume that number of dimensions are the same in torch op
        print(output)
        print("Here1")
        shape_calc_fns = [None] * args[0].ndim
        for i in range(args[0].ndim):
            input_node_expr = [syms_arg[i].node.expr for syms_arg in syms_args]
            print(f"Expected arguments: {len(tuple(input_node_expr))}")  # Should be 2

            shape_calc_fns[i] = lambdify(tuple(input_node_expr), output.shape[i].node.expr, "math")

        print("Here2")


        out_desc = args[0].like()
        for i in range(out_desc.ndim):
            input_shape_expr = [arg.shape_expr[i] for arg in args]
            print(f"actual count: {len(tuple(input_shape_expr))}")
            print(shape_calc_fns[i])
            out_desc.shape_expr[i] = shape_calc_fns[i](*input_shape_expr)
            
        print("Here3")

        return out_desc
        
    # [SOME PYTHON CODE HERE]
    codegen_plugin = f"""
{plugin_signature}
    return _generic_plugin_desc({args_input}, {kwargs_input})
    """

#     codegen_plugin = f"""
# def add_plugin_desc(a):
#     return a
#     """
    
    print(codegen_plugin)

    plugin_code = compile(codegen_plugin, "<string>", "exec")
    
    print(type(plugin_code))
    print(plugin_code.co_consts[0])
        

    globals()["_generic_plugin_desc"] = _generic_plugin_desc
    # globals()["FakeTensorMode"] = FakeTensorMode

    
    from types import FunctionType
    
    
    plugin= FunctionType(plugin_code.co_consts[0], globals(), "plugin")
    
    print(plugin)
    
    print(f"Function name: {plugin.__name__}")
    print(f"Argument count: {plugin.__code__.co_argcount}")
    print(f"Argument names: {plugin.__code__.co_varnames[:plugin.__code__.co_argcount]}")
    print(f"Function bytecode: {plugin.__code__.co_code}")
    
    plugin.__annotations__ = {'X' : trtp.TensorDesc, 'Y' : trtp.TensorDesc, 'b' : float, 'a': int, 'return': trtp.TensorDesc}
    
    trtp.register(plugin_name)(plugin)
    
    
    # plugin implementation
    # def generate_impl_signature(registered_signature):
        
    
    def _generic_plugin_impl(outputs, stream, *args, **kwargs):
        in_tensors = [
            torch.as_tensor(i, device="cuda") for i in args
        ]
        
        dest_tensors = [torch.as_tensor(o, device="cuda") for o in outputs]
        
        stream = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(stream):
            out_tensors = torch_op(*in_tensors, **kwargs)
            [d.copy_(o) for (d, o) in zip(dest_tensors, out_tensors)]

    
    plugin_impl_func = f"""
{plugin_impl_signagture}
    _generic_plugin_impl(outputs, stream, {args_input}, {kwargs_input})
    """
    
    print(plugin_impl_func)
    
    plugin_impl_code = compile(plugin_impl_func, "<string>", "exec")
    
    print(type(plugin_impl_code))
    print(plugin_impl_code.co_consts[0])
        

    globals()["_generic_plugin_impl"] = _generic_plugin_impl
    
    
    plugin_impl= FunctionType(plugin_impl_code.co_consts[0], globals(), "plugin_impl")
    
    print(plugin_impl)
    
    print(f"Function name: {plugin_impl.__name__}")
    print(f"Argument count: {plugin_impl.__code__.co_argcount}")
    print(f"Argument names: {plugin_impl.__code__.co_varnames[:plugin_impl.__code__.co_argcount]}")
    print(f"Function bytecode: {plugin_impl.__code__.co_code}")
    
    plugin_impl.__annotations__ = {'X' : trtp.Tensor, 'Y' : trtp.Tensor, 'b' : float, 'a': int, 'outputs' : Tuple[trtp.Tensor], 'stream' : int}
    
    import inspect
    sig = inspect.signature(plugin_impl)
    # registered_attr_names = plugin_def.input_attrs.keys()

    # input arg annotations are optional, but we will validate if provided
    for name, param in sig.parameters.items():
        print(name)
        print(param.annotation)
        

    trtp.impl(plugin_name)(plugin_impl)

    
    
    return plugin
    
generate_plugin("torchtrt_ex::elementwise_mul")

# @trtp.register("torchtrt_ex::elementwise_mul")
# def _(x: trtp.TensorDesc, y: trtp.TensorDesc, b: float, a: int) -> Tuple[trtp.TensorDesc]:
#     from torch._subclasses.fake_tensor import FakeTensorMode
#     from torch.fx.experimental.symbolic_shapes import ShapeEnv
#     from sympy import lambdify
#     shape_env = ShapeEnv()
#     fake_mode = FakeTensorMode(shape_env=shape_env)
#     sample_x = {f"x{i}": 5 for i in range(x.ndim)}
#     sample_y = {f"y{i}": 5 for i in range(y.ndim)}
#     syms_x = [mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC) for k,v in sample_x.items()]
#     syms_y = [mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC) for k,v in sample_y.items()]
#     with FakeTensorMode() as fake_mode:
#         fake_x = torch.randn(syms_x)
#         fake_y = torch.randn(syms_y)
#         z = torch.ops.torchtrt_ex.elementwise_mul(fake_x, fake_y, b, a)

#     shape_calc_fns = [None] * x.ndim
#     for i in range(x.ndim):
#         shape_calc_fns[i] = lambdify((syms_x[i].node.expr, syms_y[i].node.expr), z.shape[i].node.expr, "math")

#     out_desc = x.like()
#     for i in range(out_desc.ndim):
#         out_desc.shape_expr[i] = shape_calc_fns[i](x.shape_expr[i], y.shape_expr[i])
#     return out_desc


# @trtp.impl("torchtrt_ex::elementwise_mul")
# def _(X: trtp.Tensor, Y: trtp.Tensor, b: float, a: int, outputs: Tuple[trtp.Tensor], stream: int):
#     # This should be based on Torch schema
#     in_tensors = [
#         torch.as_tensor(i, device="cuda") for i in (X, Y)
#     ]  # What is the right device??
#     dest_tensors = [torch.as_tensor(o, device="cuda") for o in outputs]

#     stream = torch.cuda.ExternalStream(stream)
#     with torch.cuda.stream(stream):
#         out_tensors = torch.ops.torchtrt_ex.elementwise_mul(*in_tensors, b, a)
#         [d.copy_(o) for (d, o) in zip(dest_tensors, out_tensors)]

# @trtp.impl("torchtrt_ex::elementwise_mul")
# def _(x: trtp.Tensor, y: trtp.Tensor, b: float, a: int, outputs: Tuple[trtp.Tensor], stream: int):
#     # Define block size
#     BLOCK_SIZE = 1024

#     # Grid of programs
#     grid = lambda meta: (x.numel() // meta['BLOCK_SIZE'],)

#     x_t = torch.as_tensor(x, device="cuda")
#     y_t = torch.as_tensor(y, device="cuda")
#     z_t = torch.as_tensor(outputs[0], device="cuda")
#     # Launch the kernel
#     elementwise_mul_kernel[grid](x_t, y_t, z_t, BLOCK_SIZE=BLOCK_SIZE)

_ = torch_tensorrt.dynamo.conversion.plugins.generate_plugin_converter("torchtrt_ex::elementwise_mul", supports_dynamic_shapes=True)

from torch_tensorrt.dynamo.conversion._ConverterRegistry import DYNAMO_CONVERTERS

import torch_tensorrt as torchtrt
import tensorrt as trt
with torchtrt.logging.errors():
    model_trt = torchtrt.compile(my_model, inputs=[m, n], debug=True, min_block_size=1)
    for i in range(300):
        res = model_trt(m, n)
        print(res)
        assert torch.allclose(res, my_model(m,n))

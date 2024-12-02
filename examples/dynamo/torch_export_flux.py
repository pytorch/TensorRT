
# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import FluxPipeline, FluxTransformer2DModel
from utils import export_llm, generate
from torch.export import Dim
from typing import Optional, Dict, Any
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

import time
from contextlib import contextmanager

@contextmanager
def timer(logger, name:str):
    logger.info(f"{name} section Start...")
    start = time.time()
    yield
    end = time.time()
    logger.info(f"{name} section End...")
    logger.info(f"{name} section elapsed time: {end - start} seconds")

class MyModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor = None,
                pooled_projections: torch.Tensor = None,
                timestep: torch.LongTensor = None,
                img_ids: torch.Tensor = None,
                txt_ids: torch.Tensor = None,
                guidance: torch.Tensor = None,
                joint_attention_kwargs: Optional[Dict[str, Any]] = None,
                return_dict: bool = False, **kwargs):


        return self.module.forward(
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        # guidance,
        # joint_attention_kwargs,
        # return_dict
        )

def wrap_pipeline_transformer_call(instance, prompt, max_sequence_length):
    from unittest.mock import patch

# Assume `instance` is your class instance containing the `__call__` method

# Use patch.object to mock the __call__ method of self.transformer
    with patch.object(instance.transformer, 'forward', wraps=instance.transformer.forward) as mock_transformer_call:
        # one step is enough for intercept the inputs
        image =instance(
                prompt,
                guidance_scale=0.0,
                num_inference_steps=1,
                max_sequence_length=max_sequence_length,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]


        # Access the call arguments of the first (or specific) call
        if mock_transformer_call.call_args_list:
            args, kwargs = mock_transformer_call.call_args_list[0]
            # Store the inputs in a tuple
            intercepted_inputs = (args, kwargs)
            
            # print("Intercepted args:", args)
            # print("Intercepted kwargs:", kwargs)
            return (args, kwargs)
        else:
            print("No calls were made to self.transformer.__call__")
            return (None, None)


if __name__ == "__main__":

    # config
    dryrun = False

    # parameter setting
    batch_size = 2
    max_seq_len = 256
    prompt = ["A cat holding a sign that says hello world" for _ in range(batch_size)]
    device = "cuda:0"
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", 
                                        torch_dtype=torch.float16, num_layers=1)
    # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", 
    #                                     torch_dtype=torch.float16, device_map="balanced")
    pipe.to(device)
    # image = pipe(
    #     prompt,
    #     guidance_scale=0.0,
    #     num_inference_steps=4,
    #     max_sequence_length=256,
    #     generator=torch.Generator("cpu").manual_seed(0)
    # ).images[0]
    # image.save("pytorch_flux-schnell.png")
    # breakpoint()
    # pipe.reset_device_map()
    # pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    example_args , example_kwargs = wrap_pipeline_transformer_call(pipe, prompt, max_seq_len)
    tensor_inputs = ['hidden_states', 'timestep', 'pooled_projections', 'encoder_hidden_states', 'txt_ids', 'img_ids' ]
    example_kwargs_shapes = {key: example_kwargs[key].shape for key in tensor_inputs}
    BATCH = Dim("batch", min=1, max=batch_size)
    SEQ_LEN = Dim("seq_len", min=1, max=max_seq_len)
    dynamic_shapes = ({0 : BATCH}, 
                       {0 : BATCH,
                        1 : SEQ_LEN,
                        },
                       {0 : BATCH},
                       {0 : BATCH},
                       {0 : BATCH},
                       {0 : BATCH,
                        1 : SEQ_LEN,
                                },
                       # None,
                       # None,
                       # None,
                      )
    example_args = (
                    example_kwargs['hidden_states'], 
                    example_kwargs['encoder_hidden_states'],
                    example_kwargs['pooled_projections'],
                    example_kwargs['timestep'],
                    example_kwargs['img_ids'],
                    example_kwargs['txt_ids'],
                    # example_kwargs['guidance'],
                    # example_kwargs['joint_attention_kwargs'],
                    # example_kwargs['return_dict'],

    )

    # dynamic_shapes = {'hidden_states': {0 : BATCH}, 
    #                   'encoder_hidden_states': {0 : BATCH,
    #                                             1 : SEQ_LEN,
    #                                             },
    #                   'pooled_projections': {0 : BATCH},
    #                   'timestep': {0 : BATCH},
    #                   'img_ids': {0 : BATCH},
    #                   'txt_ids': {0 : BATCH,
    #                               1 : SEQ_LEN,
    #                             },
    #                   'guidance': None,
    #                   'joint_attention_kwargs': None,
    #                   'return_dict': None,
    #                   }

    with timer(logger=logger, name="ep_gen"):
        with torch.no_grad():

            # model = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-schnell",torch_dtype=torch.float16)
            model = MyModule(pipe.transformer).eval().half().to(device)
            # try:
            #     logger.info("Trying to export the model using torch.export.export()..")
            #     # print("Trying to export the model using torch.export.export()..")
            #     # strict=False only enables aotautograd tracing and excludes dynamo.
            #     # Have core dump this path
            #     ep = torch.export.export(
            #         model, args=example_args, kwargs=example_kwargs, dynamic_shapes=dynamic_shapes, strict=False
            #     )
            # except:
            logger.info("Directly use _export because torch.export.export doesn't work")
            # This API is used to express the constraint violation guards as asserts in the graph.
            from torch.export._trace import _export
            ep = _export(
                model,
                args=example_args, 
                # kwargs=example_kwargs, 
                dynamic_shapes=dynamic_shapes,
                strict=False,
                allow_complex_guards_as_runtime_asserts=True,
            )

    logger.info(f"Generating TRT engine now, dryrun={dryrun}...")
    # print("Generating TRT engine now...")
    #TODO: if some non-tensor input, do we still need to provide them.
    with timer(logger, "trt_gen"):
        with torch_tensorrt.logging.debug():
            trt_start = time.time()
            trt_model = torch_tensorrt.dynamo.compile(
                            ep,
                            inputs=list(example_args),
                            enabled_precisions={torch.float32},
                            truncate_double=True,
                            device=torch.device(device),
                            disable_tf32=True,
                            use_explicit_typing=True,
                            dryrun=dryrun,
                            debug=True,
                            use_fp32_acc=True,
                        )
            trt_end = time.time()

    del pipe
    del ep
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    with timer(logger, "trt_save"):
        try:
            trt_ep = torch.export.export(trt_model, args=example_args,
                                dynamic_shapes=dynamic_shapes, strict=False)
            torch.export.save(trt_ep, "trt.ep")
        except Exception as e:
            import traceback
            # Capture the full traceback
            tb = traceback.format_exc()
            logger.warning("An error occurred. Here's the traceback:")
            # print(tb)
            logger.warning(tb)
            breakpoint()
            torch_tensorrt.save(trt_model, "trt.ep")    
        # finally:
        #     breakpoint()



    # if not dryrun:
    #     pipe.transformer.forward = MyModule(trt_model).forward
    #     with timer(logger, "trt_infer"):
    #         image = pipe(
    #             prompt,
    #             guidance_scale=0.0,
    #             num_inference_steps=4,
    #             max_sequence_length=256,
    #             generator=torch.Generator("cpu").manual_seed(0)
    #         ).images[0]
    #     image.save("trt_flux-schnell.png")
        breakpoint()



                      

# breakpoint()
# flux_model_ep = export_llm(model, inputs=)

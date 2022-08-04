# Owner(s): ["oncall: gpu_enablement"]
import functools
from unittest import TestCase

import torch_tensorrt.fx.observer as ob
from test_observer import execution_verifier, set_observer_callback_rethrow
from torch_tensorrt.fx.passes.lower_basic_pass import fuse_permute_linear


class ObserverGPUTests(TestCase):
    @set_observer_callback_rethrow
    def test_observe_lowerer(self):
        """
        Test that we can observe the execution of `fuse_permute_linear` during
        lowering.
        """
        from dataclasses import replace

        import torch
        import torch.nn as nn

        import torch_tensorrt.fx.lower as lower
        from torch_tensorrt.fx.lower_setting import LowerSetting

        class Model(nn.Module):
            def forward(self, x, y):
                return x + y

        mod = Model().cuda()
        inp = [torch.rand(1, 10), torch.rand(1, 10)]
        inp = [i.cuda() for i in inp]
        mod(*inp)

        with execution_verifier() as verify_execution:

            lowerer = lower.Lowerer.create(
                lower_setting=LowerSetting(min_acc_module_size=0)
            )

            @verify_execution
            def observe_fuse_permute_linear_post(ctx: ob.ObserveContext):
                """
                Called when fuse_permute_linear is executed. Decorated with
                `verify_execution` so if this function is not executed, the
                test fails.
                """
                assert ctx.callable is fuse_permute_linear.orig_func

            # Register the observer callback and do the lowering
            with fuse_permute_linear.observers.post.add(
                observe_fuse_permute_linear_post
            ):
                lowerer(mod, inp)

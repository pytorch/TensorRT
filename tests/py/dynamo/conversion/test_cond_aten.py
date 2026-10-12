import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestCondConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("pred_true", True),
            ("pred_false", False),
        ]
    )
    def test_cond_add_sub(self, _, pred):
        class CondAddSub(nn.Module):
            def forward(self, x, predicate):
                return torch.cond(
                    predicate,
                    lambda value: value + 1,
                    lambda value: value - 1,
                    (x,),
                )

        self.run_test(
            CondAddSub(),
            [torch.randn(1, 4), torch.tensor(pred)],
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ("pred_true", True),
            ("pred_false", False),
        ]
    )
    def test_cond_one_element_pred(self, _, pred):
        class CondAddSub(nn.Module):
            def forward(self, x, predicate):
                return torch.cond(
                    predicate,
                    lambda value: value + 1,
                    lambda value: value - 1,
                    (x,),
                )

        self.run_test(
            CondAddSub(),
            [torch.randn(2, 3), torch.tensor([pred])],
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ("pred_true", True),
            ("pred_false", False),
        ]
    )
    def test_cond_multi_output(self, _, pred):
        class CondMulti(nn.Module):
            def forward(self, x, y, predicate):
                def true_fn(a, b):
                    return a + 1, b * 2

                def false_fn(a, b):
                    return a - 1, b / 2

                return torch.cond(predicate, true_fn, false_fn, (x, y))

        self.run_test(
            CondMulti(),
            [torch.randn(2, 2), torch.randn(2, 2), torch.tensor(pred)],
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ("pred_true", True),
            ("pred_false", False),
        ]
    )
    def test_cond_identity_branch(self, _, pred):
        class CondIdentity(nn.Module):
            def forward(self, x, predicate):
                return torch.cond(
                    predicate,
                    # torch.cond forbids returning an operand alias; clone is the
                    # documented workaround and still exercises a pass-through branch.
                    lambda value: value.clone(),
                    lambda value: value + 1,
                    (x,),
                )

        self.run_test(
            CondIdentity(),
            [torch.randn(3, 3), torch.tensor(pred)],
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ("pred_true", True),
            ("pred_false", False),
        ]
    )
    def test_cond_linear_outside(self, _, pred):
        class ConditionalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x, predicate):
                x = torch.relu(self.linear(x))
                return torch.cond(
                    predicate,
                    lambda value: value + 1,
                    lambda value: value - 1,
                    (x,),
                )

        self.run_test(
            ConditionalModel(),
            [torch.ones(1, 4), torch.tensor(pred)],
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ("true_true", True, True),
            ("true_false", True, False),
            ("false_true", False, True),
            ("false_false", False, False),
        ]
    )
    def test_cond_nested(self, _, pred_outer, pred_inner):
        class NestedCond(nn.Module):
            def forward(self, x, p1, p2):
                def true_fn(v, inner_pred):
                    return torch.cond(
                        inner_pred,
                        lambda a: a + 1,
                        lambda a: a + 2,
                        (v,),
                    )

                def false_fn(v, inner_pred):
                    return v - 1

                return torch.cond(p1, true_fn, false_fn, (x, p2))

        self.run_test(
            NestedCond(),
            [
                torch.randn(2, 2),
                torch.tensor(pred_outer),
                torch.tensor(pred_inner),
            ],
            use_dynamo_tracer=True,
            enable_passes=True,
        )


if __name__ == "__main__":
    run_tests()

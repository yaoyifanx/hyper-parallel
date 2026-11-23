# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Shared symmetric capacity and lossless overflow handling on CPU."""

import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.routed_experts.workspace import RingSpec, RingWorkspace
from hyper_parallel.core.multicore.modules.routed_experts import compute
from hyper_parallel.distributed.expert_parallel.moonep.fused import autograd
from hyper_parallel.distributed.expert_parallel.moonep.fused.pool import prepare_gradient_storage


class TestSharedTokenWorkspace(unittest.TestCase):
    """Exercise physical separation and actual row extents without an NPU."""

    def test_heap_accounts_for_receive_return_and_source(self):
        original = RingSpec(16, 4096)
        shared = RingSpec(16, 4096, token_rows=98304)
        self.assertEqual(shared.heap_bytes - original.heap_bytes, 2304 * 1024 * 1024)
        for rows in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                RingSpec(2, 16, token_rows=rows)

    def test_views_are_shared_across_phases_but_saved_clones_are_private(self):
        workspace = RingWorkspace.__new__(RingWorkspace)
        workspace.spec = RingSpec(2, 16, token_rows=8)
        workspace.token_received = torch.arange(128, dtype=torch.bfloat16).view(8, 16)
        workspace.token_returned = torch.zeros((8, 16), dtype=torch.bfloat16)
        received, returned = workspace.token_views(3, 5)
        saved = received.clone()
        self.assertEqual(tuple(returned.shape), (5, 16))
        workspace.token_views(8, 8)[0].zero_()
        self.assertTrue(torch.equal(received, torch.zeros_like(received)))
        self.assertTrue(torch.equal(saved, torch.arange(48, dtype=torch.bfloat16).view(3, 16)))
        self.assertFalse(received.untyped_storage().data_ptr() == returned.untyped_storage().data_ptr())
        self.assertIsNone(workspace.token_views(9, 2))
        with self.assertRaises(ValueError):
            workspace.token_views(-1, 2)

    def test_external_derivative_keeps_fp32_staging_disjoint_and_aligned(self):
        derivative = torch.empty((3, 16), dtype=torch.bfloat16)
        allocations = []

        def allocate(name, shape, dtype):
            allocations.append((name, shape, dtype))
            return torch.empty(shape, dtype=dtype)

        incoming, (send, receive) = prepare_gradient_storage(
            allocate, (3, 16), staging_elements=(3, 5), incoming=derivative)
        self.assertIs(incoming, derivative)
        self.assertEqual(send.dtype, torch.float32)
        self.assertEqual(receive.data_ptr() % 32, 0)
        self.assertGreaterEqual(receive.data_ptr(), send.data_ptr() + send.numel() * send.element_size())
        self.assertEqual(allocations[0][0], "gradient_staging")
        self.assertFalse(send.untyped_storage().data_ptr() == derivative.untyped_storage().data_ptr())

    def test_external_staging_rejects_overlapping_derivative_storage(self):
        parent = torch.empty(128, dtype=torch.bfloat16)
        derivative = parent[:48].view(3, 16)

        def allocate(name, shape, dtype):
            return parent.view(torch.float32)[:shape[0]]

        with self.assertRaisesRegex(ValueError, "overwrite the shared derivative"):
            prepare_gradient_storage(allocate, (3, 16), staging_elements=(3, 5), incoming=derivative)


class TestSharedBackwardSource(unittest.TestCase):
    """The weighted derivative belongs to dispatch, not the independent return."""

    def test_shared_return_does_not_redirect_the_dispatch_source(self):
        for consume_values, source_is_values in ((False, False), (True, True), (True, False)):
            with self.subTest(consume_values=consume_values, source_is_values=source_is_values):
                dy = torch.empty((2, 16), dtype=torch.bfloat16)
                values = torch.empty((4, 16), dtype=torch.bfloat16)
                source = values if source_is_values else torch.empty_like(values)
                shared_return = torch.empty_like(values)
                scores = torch.empty((2, 2), dtype=torch.float32)
                score_gradient = torch.empty_like(scores)
                order = torch.arange(4)
                key = namedtuple("Key", "layer")(0)
                weights = torch.empty((2, 16, 32)), torch.empty((2, 16, 16))
                executor = SimpleNamespace(layer=0, reuse_saved_values=consume_values, top_k=2)
                engine = SimpleNamespace(run_backward=lambda *args, **kwargs: kwargs["submit_return"]())
                plan = SimpleNamespace(grouped=None, compute=engine, inverse_order=order)
                runtime = Mock()
                runtime.backward_layers = set()
                runtime.fsdp_layers = {0}
                runtime.executions = {key: (executor, plan)}
                runtime.consume.return_value = (*weights, object())
                runtime.gradient_domain.complete_gradients.return_value = weights
                runtime.statistics = {}
                runtime.arena.reserved_bytes = 0
                phase = SimpleNamespace(score_gradient=score_gradient, output_gradient=dy,
                                        incoming=SimpleNamespace(source=source),
                                        returned=SimpleNamespace(value=shared_return), compute=object(),
                                        gradient=SimpleNamespace(accumulated=False), arena_token=object(), result=dy)
                ctx = SimpleNamespace(runtime=runtime, key=key, shapes=tuple(w.shape for w in weights),
                                      saved_tensors=(dy, values, values, scores, order),
                                      group_metadata=None, tokens=2)
                pending = SimpleNamespace(ready=None, wait=lambda: shared_return)
                native = SimpleNamespace(routed_experts_combine_backward=Mock(),
                                         routed_experts_combine_backward_consume=Mock())
                with patch.object(autograd, "_prepare_backward", return_value=phase), \
                     patch.object(autograd, "_submit", return_value=pending), \
                     patch.object(autograd, "combine", return_value=dy), \
                     patch.object(autograd.torch.npu, "current_stream", return_value=Mock()), \
                     patch.object(compute, "_load_native"), \
                     patch.object(compute.torch.ops, "hyper_parallel", native), \
                     patch.object(autograd, "combine_gradient", wraps=compute.combine_gradient) as combine:
                    autograd.SparseExpertFunction.backward(ctx, dy)
                actual = combine.call_args.kwargs["outputs"][0]
                self.assertIs(actual, source)
                self.assertIsNot(actual, shared_return)
                if source_is_values:
                    native.routed_experts_combine_backward_consume.assert_called_once()
                else:
                    self.assertIs(native.routed_experts_combine_backward.call_args.args[4], source)

    def test_forward_project_becomes_received_derivative_with_independent_da(self):
        """Original T-return completion precedes reusing Y for B dispatch writes."""
        project = torch.empty((3, 32), dtype=torch.bfloat16)
        buffers = {"project": project}

        def allocate(name, shape, dtype):
            if name not in buffers:
                buffers[name] = torch.empty(shape, dtype=dtype)
            self.assertEqual(tuple(buffers[name].shape), shape)
            return buffers[name]

        def check(error):
            if error is not None:
                raise error

        def exchange(source, route, workspace, **kwargs):
            return SimpleNamespace(source=source, value=kwargs["output"], ready=object())

        def prepare(derivative, state, gate, down, *, allocator, **kwargs):
            self.assertTrue(kwargs["consume_derivative"])
            da = allocator("project", (3, 16), torch.bfloat16)
            return SimpleNamespace(output=derivative, da=da, ready=object(), gradient_ready=object(),
                                   gate_gradient=torch.empty((1, 32, 32)), down_gradient=torch.empty((1, 16, 32)))

        runtime = SimpleNamespace(arena=SimpleNamespace(begin=lambda: object(), allocate=allocate, end=Mock()),
                                  workspace=SimpleNamespace(check_preflight=check),
                                  gradient_domain=SimpleNamespace(prepare_gradient_payload=Mock()))
        executor = SimpleNamespace(owner=SimpleNamespace(), token_workspace_rows=0, reuse_derivative=True,
                                   w2_before_input_grad=True, reuse_saved_values=True,
                                   home_count=1, rank=0, backward_ratio=0)
        plan = SimpleNamespace(grouped=None, route=SimpleNamespace(received_rows=3, source_rows=3, active_experts=(0,)),
                               gradient_staging_elements=(0, 0), gradient_transfer=None,
                               compute=SimpleNamespace(prepare_backward=prepare))
        x = torch.empty((3, 32), dtype=torch.bfloat16)
        z = torch.empty_like(x)
        values = torch.empty_like(x)
        scores = torch.empty((3, 1), dtype=torch.float32)
        with patch.object(autograd, "prepare_exchange", side_effect=exchange):
            phase = autograd._prepare_backward(
                x, compute.ExpertState(x, z), values, scores, [], [], executor, plan, runtime,
                ((1, 32, 32), (1, 16, 32)), None)
        self.assertIs(phase.incoming.value, project)
        self.assertIs(phase.compute.output, project)
        self.assertIs(phase.compute.da, buffers["backward_project"])
        self.assertNotEqual(phase.compute.da.data_ptr(), project.data_ptr())
        self.assertIs(phase.returned.value, values)

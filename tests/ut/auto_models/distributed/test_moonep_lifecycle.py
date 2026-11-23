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
"""MoonEP规划、显式GMM反向和真实非重入重算的CPU回归。"""

from contextlib import contextmanager
import itertools
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.distributed.expert_parallel.moonep import (
    MoonEPExecutor, configure_moonep, wrap_moonep_checkpoint,
)
from hyper_parallel.distributed.expert_parallel.moonep.checkpoint import checkpoint_route_ids
from hyper_parallel.distributed.expert_parallel.moonep.fsdp_hooks import MoonEPGradientExtension
from hyper_parallel.distributed.expert_parallel.moonep.device_planning import (
    candidate_metrics, expand_segments, select_device_allocation,
)
from hyper_parallel.distributed.expert_parallel.moonep.numerics import (
    expert_backward_values, expert_forward_values, grouped_values, grouped_weight_grad,
)
from hyper_parallel.distributed.expert_parallel.moonep import checkpoint as moonep_checkpoint, planning
from hyper_parallel.distributed.expert_parallel.moonep import runtime as moonep_runtime
from hyper_parallel.distributed.expert_parallel.moonep.planning import compile_plan
from tests.common.mark_utils import arg_mark
from tests.common.moonep.planning import cpu_plan_metrics


class _Experts(torch.nn.Module):
    """真实参数边界的微型专家；例如两组8×8矩阵。"""

    def __init__(self):
        """随机初始化，不从文件加载；例如每个专家具有gate/up/down三个投影。"""
        super().__init__()
        self.gate_up = torch.nn.Parameter(torch.randn(2, 8, 16) * 0.1)
        self.down = torch.nn.Parameter(torch.randn(2, 8, 8) * 0.1)
        self._moonep_executor = MoonEPExecutor(
            self, lambda: (self.gate_up, self.down), ep_group=None, ep_size=1, num_experts=2, top_k=2,
        )

    def forward(self, x, ids, scores):
        """进入实际模块边界；例如5个token展开为10条选择。"""
        return self._moonep_executor(x, ids, scores)


class _Block(torch.nn.Module):
    """一个checkpoint区域只有一次专家调用；例如三层顺序模型中的中间层。"""

    def __init__(self):
        """建立router、专家和输出线性；例如所有参数都需要训练。"""
        super().__init__()
        self.router = torch.nn.Linear(8, 2)
        self.experts = _Experts()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        """可微路由与残差；例如重放使用相同ID但重新计算score梯度。"""
        probability = self.router(x).softmax(-1)
        ids = checkpoint_route_ids(self.router, probability, 2)
        return self.linear(self.experts(x, ids, probability.gather(-1, ids))) + x


class TestMoonEPLifecycle(unittest.TestCase):
    """不初始化分布式、不依赖NPU，检查真实autograd与checkpoint。"""

    def test_replay_context_does_not_read_weights_before_module_hooks(self):
        """Replay restores private resources while module hooks retain parameter ownership."""
        identity, key = object(), object()
        lifecycle = SimpleNamespace(prepare=mock.Mock(side_effect=AssertionError("early weight read")),
                                    prepare_backward_resources=mock.Mock(),
                                    statistics={"recomputations": 0})
        executor = SimpleNamespace(runtime=lifecycle)
        tape = moonep_checkpoint._Tape(execution=(executor, key), identity=identity)
        phase = SimpleNamespace(is_recomputing=True, invocation_id=identity)
        with mock.patch.object(moonep_checkpoint, "get_recompute_state", return_value=phase):
            with moonep_checkpoint._phase(torch.nn.Module(), tape, True):
                lifecycle.prepare_backward_resources.assert_called_once_with(key)
                self.assertIs(moonep_checkpoint.bind_execution(executor, mock.Mock()), key)
        lifecycle.prepare.assert_not_called()
        self.assertEqual(lifecycle.statistics["recomputations"], 1)

    def test_checkpoint_single_execution_error_contract(self) -> None:
        """Keep zero-call forward, one-call replay and all single-record rejection boundaries."""
        identity, key = object(), object()
        state = SimpleNamespace(is_recomputing=False, invocation_id=identity)
        executor = SimpleNamespace(runtime=SimpleNamespace(statistics={"recomputations": 0}))
        block, tape = torch.nn.Module(), moonep_checkpoint._Tape()
        factory = mock.Mock(return_value=key)
        previous = moonep_checkpoint._SCOPE.get()
        with mock.patch.object(moonep_checkpoint, "get_recompute_state", return_value=state):
            with moonep_checkpoint._phase(block, tape, False):
                self.assertIs(moonep_checkpoint.bind_execution(executor, factory), key)
                with self.assertRaisesRegex(ValueError, "仅执行一次"):
                    moonep_checkpoint.bind_execution(executor, factory)
            factory.assert_called_once_with()
            state.is_recomputing = True
            with moonep_checkpoint._phase(block, tape, True):
                self.assertIs(moonep_checkpoint.bind_execution(executor, factory), key)
                with self.assertRaisesRegex(RuntimeError, "不存在"):
                    moonep_checkpoint.bind_execution(executor, factory)
            factory.assert_called_once_with()
            with self.assertRaisesRegex(RuntimeError, "没有执行"):
                with moonep_checkpoint._phase(block, tape, True):
                    pass
            with moonep_checkpoint._phase(block, tape, True):
                with self.assertRaisesRegex(RuntimeError, "顺序发生变化"):
                    moonep_checkpoint.bind_execution(object(), factory)
                with self.assertRaisesRegex(RuntimeError, "不存在"):
                    moonep_checkpoint.bind_execution(executor, factory)
            state.invocation_id = object()
            with self.assertRaisesRegex(RuntimeError, "原始单专家"):
                with moonep_checkpoint._phase(block, tape, True):
                    pass
            state.is_recomputing = False
            empty = moonep_checkpoint._Tape()
            with moonep_checkpoint._phase(block, empty, False):
                pass
            self.assertIsNone(empty.execution)
            state.is_recomputing = True
            with self.assertRaisesRegex(RuntimeError, "原始单专家"):
                with moonep_checkpoint._phase(block, empty, True):
                    pass
        self.assertIs(moonep_checkpoint._SCOPE.get(), previous)

    def test_nested_checkpoint_scope_restores_after_normal_and_error_exit(self) -> None:
        """Inner tapes own one call and always restore the outer ContextVar token."""
        previous = moonep_checkpoint._SCOPE.get()
        for fail in (False, True):
            with self.subTest(inner_error=fail):
                parent_id, child_id = object(), object()
                state = SimpleNamespace(is_recomputing=False, invocation_id=parent_id)
                outer, inner = moonep_checkpoint._Tape(), moonep_checkpoint._Tape()
                with mock.patch.object(moonep_checkpoint, "get_recompute_state", return_value=state):
                    with moonep_checkpoint._phase(torch.nn.Module(), outer, False):
                        state.invocation_id = child_id
                        try:
                            with moonep_checkpoint._phase(torch.nn.Module(), inner, False):
                                moonep_checkpoint.bind_execution(object(), object)
                                if fail:
                                    raise ValueError("nested checkpoint error")
                        except ValueError as error:
                            self.assertTrue(fail)
                            self.assertEqual(str(error), "nested checkpoint error")
                        self.assertIs(moonep_checkpoint._SCOPE.get().tape, outer)
                        state.invocation_id = parent_id
                        moonep_checkpoint.bind_execution(object(), object)
                self.assertIsNotNone(inner.execution)
                self.assertIsNotNone(outer.execution)
                self.assertIs(moonep_checkpoint._SCOPE.get(), previous)


    def test_native_backward_prefetch_submits_ep_extension_after_ag(self):
        """保留原生 AG 并按顺序接 EP；例如 B7 的预取先提交 B6 AG 再执行扩展。"""
        order = []
        targets = [SimpleNamespace(hsdp_state=SimpleNamespace(
            module_name=f'expert{index}',
            prefetch=lambda index=index: order.append(('efs_ag', index)),
            complete_prefetch=lambda index=index: order.append(('complete', index)))) for index in range(2)]
        scheduler = object.__new__(HSDPSchedulerV2)
        scheduler.hsdp_state = SimpleNamespace(module_name='current',
                                              unshard=lambda: order.append(('restore_current',)))
        scheduler.reshard_after_forward = False
        scheduler.backward_prefetch_cells = [SimpleNamespace(hsdp_scheduler=target) for target in targets]
        scheduler._hsdp_backward_pre_hook(None, None)
        self.assertEqual(order, [('restore_current',), ('efs_ag', 0), ('efs_ag', 1),
                                 ('complete', 0), ('complete', 1)])


    def test_gradient_extension_waits_only_before_fsdp_reduce(self):
        """先发起 BF16 副本回传，参数累积后再等待；例如转置 gate 的第 1 行回 home。"""
        gate_param = torch.nn.Parameter(torch.zeros(2, 4, 2, dtype=torch.bfloat16))
        down_param = torch.nn.Parameter(torch.zeros(2, 2, 2, dtype=torch.bfloat16))
        executor = SimpleNamespace(
            read_weights=lambda: (gate_param.transpose(1, 2), down_param), gradient_group=object(),
        )
        runtime = configure_moonep(_Experts())
        extension = MoonEPGradientExtension(executor, runtime)
        scheduler = SimpleNamespace(hsdp_state=SimpleNamespace(hsdp_params=[
            SimpleNamespace(unsharded_param=gate_param), SimpleNamespace(unsharded_param=down_param),
        ]))
        plan = SimpleNamespace(
            home_count=2, weight_send=[1], weight_recv=[1],
            send_ids=torch.tensor([1]), any_migration=True,
        )
        gate_grad = torch.arange(24, dtype=torch.bfloat16).reshape(3, 2, 4)
        down_grad = torch.arange(12, dtype=torch.bfloat16).reshape(3, 2, 2)
        waits = []

        def exchange(output, source, *_args, **_kwargs):
            output.copy_(source)
            return SimpleNamespace(wait=lambda: waits.append(True))

        with mock.patch.object(moonep_runtime.dist, 'all_to_all_single', side_effect=exchange):
            local_gate, local_down = extension.start(gate_grad, down_grad, plan)
            self.assertEqual(waits, [])
            gate_param.grad = local_gate.transpose(1, 2).contiguous()
            down_param.grad = local_down.clone()
            extension.complete(scheduler.hsdp_state)
        self.assertEqual(len(waits), 1)
        self.assertEqual(extension.pending, [])
        torch.testing.assert_close(gate_param.grad[1], (gate_grad[1] + gate_grad[2]).transpose(0, 1))
        torch.testing.assert_close(down_param.grad[1], down_grad[1] + down_grad[2])
        torch.testing.assert_close(gate_param.grad[0], gate_grad[0].transpose(0, 1))

    def test_weight_table_stream_ownership_and_consumer_lifetime(self):
        """Persistent W tables keep ready/read fences and C lifetime protection."""
        expert = _Experts()
        runtime = configure_moonep(expert)
        weight, compute, ready = mock.Mock(), mock.Mock(), object()
        runtime.stream = weight
        events = []
        new_empty = torch.Tensor.new_empty

        @contextmanager
        def _stream_context(stream):
            self.assertIs(stream, weight)
            events.append("allocate_on_w")
            yield
            events.append("leave_w")

        def _allocate(tensor, shape):
            self.assertEqual(events[-1], "allocate_on_w")
            return new_empty(tensor, shape)

        def _record(_tensor, stream):
            self.assertIs(stream, compute)
            events.append("protect_c")

        compute.wait_event.side_effect = lambda event: events.append(("wait_ready", event))
        npu = SimpleNamespace(stream=_stream_context, current_stream=lambda: compute, Event=mock.Mock)
        with mock.patch.object(torch, "npu", npu, create=True),\
                mock.patch.object(torch.Tensor, "new_empty", _allocate):
            slot = runtime._new_slot(expert.gate_up, expert.down)
        slot.gate_up.fill_(3)
        slot.down.fill_(5)
        key = object()
        lease = SimpleNamespace(slot=slot, gate_up=slot.gate_up[:1], down=slot.down[:1], ready=ready,
                                 consumed=False)
        runtime.weight_leases[key] = lease
        with mock.patch.object(torch, "npu", npu, create=True),\
                mock.patch.object(runtime, "prepare", side_effect=AssertionError("consumer launched preparation")),\
                mock.patch.object(torch.Tensor, "record_stream", _record):
            gate_up, down = moonep_runtime.MoonEPRuntime.consume(runtime, key)
            runtime.release(key)
        self.assertEqual(events, ["allocate_on_w", "leave_w", ("wait_ready", ready), "protect_c", "protect_c"])
        self.assertEqual(runtime.weight_leases, {})
        slot.compute_read_done.record.assert_called_once_with(compute)
        torch.testing.assert_close(gate_up, torch.full_like(gate_up, 3))
        torch.testing.assert_close(down, torch.full_like(down, 5))

    def test_weight_workspace_is_bounded_and_reused(self):
        """AG与EP按需共池且增长不破坏旧视图；例如预算8元素时9元素不能扩池。"""
        expert = _Experts()
        runtime = configure_moonep(expert)
        template = torch.empty(1, dtype=torch.bfloat16)
        first = runtime._borrow_weight_workspace("send", template, 4, 8)
        first.fill_(3)
        self.assertEqual(runtime.weight_workspace["send"].numel(), 4)
        second = runtime._borrow_weight_workspace("send", template, 3, 8)
        self.assertEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(second, torch.full((3,), 3, dtype=torch.bfloat16))
        grown = runtime._borrow_weight_workspace("send", template, 7, 8)
        self.assertNotEqual(grown.data_ptr(), first.data_ptr())
        self.assertEqual(runtime.weight_workspace["send"].numel(), 8)
        torch.testing.assert_close(first, torch.full((4,), 3, dtype=torch.bfloat16))
        oversized = runtime._borrow_weight_workspace("send", template, 9, 8)
        self.assertNotEqual(oversized.data_ptr(), grown.data_ptr())
        self.assertEqual(runtime.weight_workspace["send"].numel(), 8)
        empty = runtime._borrow_weight_workspace("send", template, 0, 8)
        self.assertEqual(empty.numel(), 0)
        received = runtime._borrow_weight_workspace("recv", template, 2, 8)
        self.assertEqual(runtime.weight_workspace["recv"].numel(), 2)
        self.assertNotEqual(received.untyped_storage().data_ptr(), grown.untyped_storage().data_ptr())
        runtime.finish_step()
        self.assertEqual(runtime.weight_workspace, {})

    def test_oversized_weight_send_uses_the_ep_stream(self):
        """Oversized sends share EP allocation ordering and retain the collective wait."""
        expert = _Experts()
        gate_up = expert.gate_up.transpose(1, 2).contiguous().transpose(1, 2)
        down = expert.down.transpose(1, 2)
        expert._moonep_executor.read_weights = lambda: (gate_up, down)
        runtime = configure_moonep(expert)
        key = moonep_runtime.ExecutionKey(0, 0, runtime.version)
        plan = SimpleNamespace(any_migration=True, home_count=2, weight_send=[3], weight_recv=[1],
                               send_ids=torch.tensor([0, 1, 0]), group_counts=[1, 1, 1])
        runtime.executions[key] = expert._moonep_executor, plan
        runtime.prepare_home_layout(expert._moonep_executor, key)
        compute, ep = object(), mock.Mock()
        runtime.stream = ep
        active, events, recorded = [compute], [], []
        original_empty = torch.Tensor.new_empty

        @contextmanager
        def stream_context(stream):
            previous, active[0] = active[0], stream
            try:
                yield
            finally:
                active[0] = previous

        def allocate(tensor, shape):
            self.assertIs(active[0], ep)
            events.append("allocate")
            return original_empty(tensor, shape)

        def exchange(output, source, *_args, **_kwargs):
            self.assertIs(active[0], ep)
            events.append("submit")
            output.copy_(source[:1])
            return SimpleNamespace(wait=lambda: events.append("wait"))

        npu = SimpleNamespace(stream=stream_context, current_stream=lambda: active[0], Event=mock.Mock)
        with mock.patch.object(torch, "npu", npu, create=True),\
                mock.patch.object(torch.Tensor, "new_empty", allocate),\
                mock.patch.object(torch.Tensor, "record_stream", lambda value, _stream: recorded.append(value)),\
                mock.patch.object(moonep_runtime.dist, "all_to_all_single", side_effect=exchange):
            lease = runtime.prepare(key)
        self.assertEqual(events[-2:], ["submit", "wait"])
        self.assertNotIn("send", runtime.weight_workspace)
        ep.wait_stream.assert_called_once_with(compute)
        lease.ready.record.assert_called_once_with(ep)
        self.assertFalse(any(value is gate_up or value is down for value in recorded))
        self.assertTrue(any(value is lease.slot.home_gate_up for value in recorded))
        self.assertTrue(any(value is lease.slot.home_down for value in recorded))
        torch.testing.assert_close(lease.gate_up[:2], gate_up)
        torch.testing.assert_close(lease.gate_up[2], gate_up[0])
        torch.testing.assert_close(lease.down[2], down[0])


    def test_receive_order_preserves_source_and_row_order(self):
        """分段平移等价于接收标签的稳定排序；例如交错来源和零长段也保持原行次序。"""
        generator = torch.Generator().manual_seed(923)
        for ranks, experts in ((1, 1), (2, 2), (4, 16), (8, 128)):
            for sample in range(12):
                counts = torch.randint(0, 9, (ranks, experts), generator=generator)
                if sample == 0:
                    counts.zero_()
                starts = (counts.reshape(-1).cumsum(0) - counts.reshape(-1)).reshape_as(counts)
                labels = [expert for row in counts.tolist() for expert, count in enumerate(row)
                          for _ in range(count)]
                expected = torch.tensor(sorted(range(len(labels)), key=labels.__getitem__), dtype=torch.long)
                actual = expand_segments(starts.T, counts.T, len(labels))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def setUp(self):
        """CPU测试固定checkpoint默认设备；例如机器安装torch_npu也不初始化NPU。"""
        self.previous_device = DefaultDeviceType.get_device_type()
        DefaultDeviceType.set_device_type('cpu')

    def tearDown(self):
        """恢复全局默认值；例如后续NPU测试仍使用其原始设置。"""
        DefaultDeviceType.set_device_type(self.previous_device)

    def _run_model(self, recompute, depth, evaluate=False):
        """执行连续两步并收集完整梯度；例如eval→train不能残留执行键。"""
        torch.manual_seed(45)
        model = torch.nn.Sequential(*[_Block() for _ in range(3)])
        if recompute:
            for index, block in enumerate(model):
                model[index] = wrap_moonep_checkpoint(block)
        runtime = configure_moonep(model, prefetch_depth=depth)
        inputs = torch.randn(5, 8, requires_grad=True)
        if evaluate:
            with torch.no_grad():
                model(inputs)
            self.assertEqual(runtime.trace, [])
            self.assertEqual(runtime.executions, {})
        records = []
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            inputs.grad = None
            loss = model(inputs).square().sum()
            loss.backward()
            runtime.finish_step()
            records.append([loss.detach(), inputs.grad.clone(), *[p.grad.clone() for p in model.parameters()]])
        self.assertEqual(runtime.version, 2)
        self.assertEqual(runtime.trace, [])
        self.assertEqual(runtime.weight_leases, {})
        self.assertEqual(runtime.statistics['plans'], 9 if evaluate else 6)
        self.assertEqual(runtime.statistics['recomputations'], 6 if recompute else 0)
        self.assertLessEqual(runtime.statistics['max_live_weight_leases'], 2)
        self.assertEqual(len(runtime.slots), runtime.statistics['max_live_weight_leases'])
        return records

    def test_checkpoint_prefetch_and_eval_train_match(self):
        """四种生命周期组合与评估切换数值一致；例如R2不能再次规划或重复交换权重。"""
        expected = self._run_model(False, 0)
        for checkpoint, depth in itertools.product((False, True), (0, 1)):
            with self.subTest(checkpoint=checkpoint, depth=depth):
                actual = self._run_model(checkpoint, depth, evaluate=True)
                for step_a, step_b in zip(actual, expected):
                    for value, reference in zip(step_a, step_b):
                        torch.testing.assert_close(value, reference, rtol=1e-6, atol=1e-6)
        # 独立脚本路径只预取紧邻前一专家；较大的旧 depth 值不再预留空槽。
        self._run_model(False, 4)

    def test_explicit_backward_matches_independent_autograd(self):
        """比较dX、两组dW及前向；例如限幅端点和空专家都经过独立torch公式。"""
        for activation, limit in (('clipped', 0.0), ('clipped', 0.3), ('swiglu', 0.0)):
            with self.subTest(activation=activation, limit=limit):
                torch.manual_seed(9)
                x = torch.randn(5, 4, requires_grad=True)
                gate_up = torch.randn(3, 4, 6, requires_grad=True)
                down = torch.randn(3, 3, 4, requires_grad=True)
                boundaries = torch.tensor([2, 2, 5])
                outputs = []
                begin = 0
                for expert, end in enumerate(boundaries.tolist()):
                    left, right = (x[begin:end] @ gate_up[expert]).chunk(2, -1)
                    if limit > 0:
                        left, right = left.clamp(max=limit), right.clamp(-limit, limit)
                    outputs.append((torch.nn.functional.silu(left) * right) @ down[expert])
                    begin = end
                expected = torch.cat(outputs)
                gradient = torch.randn_like(expected)
                derivatives = torch.autograd.grad(expected, (x, gate_up, down), gradient)
                actual, saved = expert_forward_values(x, gate_up, down, boundaries, limit, grouped_values, activation)
                gradients = expert_backward_values(gradient, saved, gate_up, down, grouped_values, grouped_weight_grad)
                torch.testing.assert_close(actual, expected)
                for value, reference in zip(gradients, derivatives):
                    torch.testing.assert_close(value, reference, atol=2e-6, rtol=2e-6)

    def test_executor_permutation_does_not_save_computed_storage(self):
        """真实执行器恢复排列只保存索引；例如10条交错路由不会多保留一份[10,8]专家输出。"""
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(73)
                model = _Experts().to(dtype=dtype)
                runtime = configure_moonep(model, prefetch_depth=0)
                x = torch.randn(5, 8, dtype=dtype, requires_grad=True)
                ids = torch.tensor([[1, 0], [0, 1], [1, 0], [1, 0], [0, 1]])
                scores = torch.rand(5, 2, dtype=dtype, requires_grad=True)
                saved_storages, computed_storages = [], []
                original_compute = moonep_runtime.expert_forward_values

                def capture_computed(*args):
                    """Identify storage without retaining it or confusing allocator address reuse."""
                    result = original_compute(*args)
                    computed_storages.append(StorageWeakRef(result[0].untyped_storage()))
                    return result

                def capture_saved(tensor):
                    """观察整个执行器的autograd保存；例如score求导需要恢复后的输出，但不需要原排序输出。"""
                    saved_storages.append(StorageWeakRef(tensor.untyped_storage()))
                    return tensor

                with mock.patch.object(moonep_runtime, 'expert_forward_values', side_effect=capture_computed),\
                        torch.autograd.graph.saved_tensors_hooks(capture_saved, lambda value: value):
                    actual = model(x, ids, scores)
                self.assertEqual(len(computed_storages), 1)
                self.assertNotIn(computed_storages[0], saved_storages)

                # 独立逐token公式同时检查恢复排列、输入梯度、router梯度和两组专家参数梯度。
                expected_rows = []
                for token in range(x.shape[0]):
                    values = []
                    for slot in range(ids.shape[1]):
                        expert = int(ids[token, slot])
                        gate_up, up = (x[token] @ model.gate_up[expert]).float().chunk(2, -1)
                        intermediate = (torch.nn.functional.silu(gate_up) * up).to(dtype)
                        values.append((intermediate @ model.down[expert]) * scores[token, slot])
                    expected_rows.append(values[0] + values[1])
                expected = torch.stack(expected_rows)
                gradient = torch.randn_like(actual)
                parameters = (x, scores, model.gate_up, model.down)
                expected_gradients = torch.autograd.grad(expected, parameters, gradient)
                actual_gradients = torch.autograd.grad(actual, parameters, gradient)
                tolerance = 3e-3 if dtype == torch.bfloat16 else 2e-6
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                for value, reference in zip(actual_gradients, expected_gradients):
                    torch.testing.assert_close(value, reference, rtol=tolerance, atol=tolerance)
                runtime.finish_step()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_bfloat16_output_with_float32_router_scores(self):
        """Feature: MoonEP mixed precision routing.
        Description: Run BF16 experts with FP32 router scores on CPU.
        Expectation: Output keeps BF16 dtype and both gradient paths are finite.
        """
        torch.manual_seed(91)
        model = _Experts().to(dtype=torch.bfloat16)
        runtime = configure_moonep(model, prefetch_depth=0)
        hidden = torch.randn(5, 8, dtype=torch.bfloat16, requires_grad=True)
        indices = torch.tensor([[1, 0], [0, 1], [1, 0], [1, 0], [0, 1]])
        scores = torch.rand(5, 2, dtype=torch.float32, requires_grad=True)

        output = model(hidden, indices, scores)
        self.assertEqual(output.dtype, torch.bfloat16)
        output.float().sum().backward()
        for name, gradient in (("hidden", hidden.grad), ("scores", scores.grad),
                               ("gate", model.gate_up.grad), ("down", model.down.grad)):
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all().item(), name)
        runtime.finish_step()


class TestMoonEPCapacity(unittest.TestCase):
    """验证默认规划合法性、迁移证书与通信元数据守恒。"""


    def test_planner_uses_bounded_fill_fallback_without_solver(self):
        """四卡容量难例由填满接收卡兜底；例如迁移 12 条而不进入整数规划。"""
        totals = torch.tensor([0, 0, 4, 4, 4, 4, 4, 4])
        counts = totals.repeat(4, 1) // 4
        _, allocation, used_fallback = planning.select_allocation(counts, 2)
        self.assertTrue(used_fallback)
        expected = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 0], [2, 2, 0, 0],
                                 [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 2, 2], [0, 0, 0, 4]])
        torch.testing.assert_close(allocation, expected, rtol=0, atol=0)
        self.assertEqual(cpu_plan_metrics(allocation.tolist(), totals.tolist(), [2] * 4)[1], 12)

    def test_compiled_dispatch_and_weight_splits_are_reciprocal(self):
        """每条路由只发送一次且权重收发互逆；例如热e0被分到两张卡。"""
        routes = [torch.tensor([0, 0, 0, 0, 1, 2]), torch.tensor([0, 0, 0, 2, 3, 3])]
        counts = torch.stack([torch.bincount(ids, minlength=4) for ids in routes])
        _, allocation, _ = planning.select_allocation(counts, 2)
        plans = [compile_plan(counts, allocation, ids, rank) for rank, ids in enumerate(routes)]
        for target, plan in enumerate(plans):
            self.assertEqual(plan.unpermute_indices.dtype, torch.int32)
            restored = plan.dispatch_order.index_select(0, plan.unpermute_indices.long())
            torch.testing.assert_close(restored, torch.arange(routes[target].numel()))
            labels = []
            for source, source_plan in enumerate(plans):
                self.assertEqual(plan.token_recv[source], source_plan.token_send[target])
                self.assertEqual(plan.weight_recv[source], source_plan.weight_send[target])
                begin = sum(source_plan.token_send[:target])
                order = source_plan.dispatch_order[begin:begin + source_plan.token_send[target]]
                labels.extend(routes[source][order].tolist())
            sorted_labels = torch.tensor(labels)[plan.sort_order]
            self.assertEqual(torch.bincount(sorted_labels, minlength=4).tolist(), allocation[:, target].tolist())
            home_count = allocation.shape[0] // len(plans)
            home = list(range(target * home_count, (target + 1) * home_count))
            active = home + [expert for expert in range(allocation.shape[0])
                             if expert not in home and allocation[expert, target] > 0]
            self.assertEqual(sorted_labels.tolist(),
                             [expert for expert in active for _ in range(int(allocation[expert, target]))])


    def test_tensor_plan_obeys_capacity_and_certificate(self) -> None:
        """Validate the production planner using independent scalar invariants."""
        generator = torch.Generator().manual_seed(8921)
        for ranks in (2, 4, 8, 16):
            for sample in range(10):
                counts = torch.randint(0, 48, (ranks, ranks * 2), generator=generator)
                if sample == 0:
                    counts.zero_()
                elif sample == 1:
                    counts[:, 1:] = 0
                returned, allocation, fallback = planning.select_allocation(counts, 2)
                torch.testing.assert_close(returned, counts, rtol=0, atol=0)
                tensor_allocation, status = select_device_allocation(counts, 2)
                torch.testing.assert_close(tensor_allocation, allocation, rtol=0, atol=0)
                self.assertEqual(bool(status[-1]), fallback)
                totals = counts.sum(0)
                metrics = cpu_plan_metrics(allocation.tolist(), totals.tolist(), [2] * ranks)
                self.assertEqual(metrics[0], 1)
                self.assertGreaterEqual(metrics[1], metrics[4])
                actual = candidate_metrics(allocation[None], totals, 2)[0]
                torch.testing.assert_close(actual, torch.tensor(metrics), rtol=0, atol=0)
                if not fallback:
                    self.assertEqual(metrics[5], 1)
        with self.assertRaisesRegex(RuntimeError, "B=0"):
            planning.select_allocation(torch.tensor([[8, 0], [0, 0]]), 0)

    def test_local_sort_precedes_count_wait_and_is_reused(self) -> None:
        """Keep counts unreadable until wait; overlap sorting without changing routing."""
        ids = torch.tensor([3, 0, 3, 1, 2, 0])
        counts = torch.bincount(ids, minlength=4).repeat(2, 1)
        _, allocation, _ = planning.select_allocation(counts, 2)
        expected = compile_plan(counts, allocation, ids, 0)
        executor = object.__new__(MoonEPExecutor)
        executor.experts, executor.ranks, executor.rank = 4, 2, 0
        executor.replica_capacity, executor.group, executor.chunked = 2, object(), None
        executor.token_dedup = False
        executor.runtime = SimpleNamespace(planning_graphs={}, statistics=dict(
            plans=0, planner_fallbacks=0, moved_entries=0, peak_copies=0, remote_experts_max=0,
            max_selection_rows=0, max_received_rows=0))
        stages = []

        def gather(output, payload, *, group, async_op):
            self.assertIs(group, executor.group)
            self.assertTrue(async_op)
            stages.append("submit")
            output.fill_(-1)

            def wait():
                self.assertEqual(stages, ["submit", "sort"])
                output.view(2, -1).copy_(payload.expand(2, -1))
                stages.append("wait")

            return SimpleNamespace(wait=wait)

        with mock.patch.object(planning, "prepare_entry_order", wraps=planning.prepare_entry_order) as sort:
            def sort_local(values, experts):
                stages.append("sort")
                return sort(values, experts)

            with mock.patch.object(moonep_runtime.dist, "all_gather_into_tensor", side_effect=gather) as submit:
                with mock.patch.object(moonep_runtime, "prepare_entry_order", side_effect=sort_local):
                    actual = executor._plan(ids)
            submit.assert_called_once()
            sort.assert_called_once()
        self.assertEqual(stages, ["submit", "sort", "wait"])
        for name in ("dispatch_order", "unpermute_indices", "sort_order", "boundaries", "send_ids"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)


    def test_fixed_home_prefix_matches_global_weights_without_output_reallocation(self):
        """独立全局权重对照固定 home 前缀及副本后缀。

        输入为随机专家分配和全局权重，期望输出直接按全局专家号索引得到。
        接收权重故意使用打包缓冲的非连续视图，同时验证最终连续槽切片不换存储。
        """
        generator = torch.Generator().manual_seed(72)
        for ranks in (1, 2, 4, 8):
            experts = ranks * 3
            for _ in range(30):
                allocation = torch.randint(0, 5, (experts, ranks), generator=generator)
                counts = torch.zeros(ranks, experts, dtype=torch.long)
                counts[0] = allocation.sum(1)
                global_gate = torch.randn(experts, 4, 6, generator=generator)
                global_down = torch.randn(experts, 3, 4, generator=generator)
                for rank in range(ranks):
                    ids = torch.repeat_interleave(torch.arange(experts), counts[rank])
                    plan = compile_plan(counts, allocation, ids, rank)
                    home = torch.arange(rank * 3, (rank + 1) * 3)
                    remote = torch.tensor(
                        [expert for expert in range(experts)
                         if expert // 3 != rank and allocation[expert, rank] > 0],
                        dtype=torch.long,
                    )
                    active = torch.cat((home, remote))
                    home_gate_up, home_down = global_gate[home], global_down[home]
                    packed = torch.cat((global_gate[remote].flatten(1),
                                        global_down[remote].flatten(1)), dim=1)
                    remote_gate_up = packed[:, :24].reshape(len(remote), 4, 6)
                    remote_down = packed[:, 24:].reshape(len(remote), 3, 4)
                    target_gate_up = torch.empty(experts + 2, 4, 6)
                    target_down = torch.empty(experts + 2, 3, 4)
                    pointers = target_gate_up.data_ptr(), target_down.data_ptr()
                    target_gate_up[:3].copy_(home_gate_up)
                    target_down[:3].copy_(home_down)
                    target_gate_up[3:len(active)].copy_(remote_gate_up)
                    target_down[3:len(active)].copy_(remote_down)
                    self.assertEqual((target_gate_up.data_ptr(), target_down.data_ptr()), pointers)
                    torch.testing.assert_close(target_gate_up[:len(active)], global_gate[active], rtol=0, atol=0)
                    torch.testing.assert_close(target_down[:len(active)], global_down[active], rtol=0, atol=0)


class TestMoonEPOptimizations(unittest.TestCase):
    """独立列表算法及内存布局反例；例如相同的专家并列时仍必须选择较小编号。"""

    def test_home_layout_reservation_reuses_scratch_and_cancels_failed_plan(self):
        """Retain one scratch lease, refresh the next version and cancel before consumption."""
        gate_up = torch.arange(256.).reshape(2, 16, 8).transpose(1, 2)
        down = torch.arange(128.).reshape(2, 8, 8).transpose(1, 2)
        owner = _Experts()
        executor = owner._moonep_executor
        executor.read_weights = lambda: (gate_up, down)
        lifecycle = configure_moonep(owner)
        key = lifecycle.new_key(executor)
        lifecycle.prepare_home_layout(executor, key)
        self.assertEqual(lifecycle.home_layout_slots, {})
        plan = SimpleNamespace(group_counts=(1, 2, 3), home_count=2,
                               weight_send=(2, 0), weight_recv=(0, 1))
        lifecycle.executions[key] = executor, plan
        lifecycle.prepare_home_layout(executor, key)
        slot = lifecycle.home_layout_slots[key]
        pointers = slot.home_gate_up.data_ptr(), slot.home_down.data_ptr()
        lifecycle.prepare_home_layout(executor, key)
        self.assertEqual(lifecycle.statistics['home_layout_preparations'], 1)
        with mock.patch.object(lifecycle, '_exchange_weights', return_value=(gate_up[:1], down[:1], [])):
            lease = lifecycle.prepare(key)
        self.assertIs(lease.slot, slot)
        self.assertEqual(lifecycle.home_layout_slots, {})
        self.assertEqual(lifecycle.statistics['home_layout_preparations'], 1)
        torch.testing.assert_close(lease.gate_up[:2], gate_up, rtol=0, atol=0)
        lifecycle.consume(key)
        lifecycle.release(key)
        lifecycle.complete(key)
        lifecycle.finish_step()
        gate_up.add_(3)
        down.add_(4)
        updated = lifecycle.new_key(executor)
        lifecycle.executions[updated] = executor, plan
        lifecycle.prepare_home_layout(executor, updated)
        self.assertEqual((slot.home_gate_up.data_ptr(), slot.home_down.data_ptr()), pointers)
        self.assertEqual(slot.home_gate_up.untyped_storage().data_ptr(),
                         slot.gate_up.untyped_storage().data_ptr())
        torch.testing.assert_close(slot.home_gate_up, gate_up, rtol=0, atol=0)
        torch.testing.assert_close(slot.home_down, down, rtol=0, atol=0)
        lifecycle.cancel_home_layout(updated)
        self.assertEqual(lifecycle.home_layout_slots, {})
        self.assertIsNone(slot.key)

    def test_replica_pack_reuses_home_table_without_delaying_receivers(self):
        """Reuse one layout conversion while receivers submit before home copies."""
        for dtype, sending, transposed in itertools.product((torch.float32, torch.bfloat16),
                                                           (False, True), (False, True)):
            with self.subTest(dtype=dtype, sending=sending, transposed=transposed):
                gate_up = torch.arange(256, dtype=dtype).reshape(2, 16, 8).transpose(1, 2)
                down = torch.arange(128, dtype=dtype).reshape(2, 8, 8).transpose(1, 2)
                if not transposed:
                    gate_up, down = gate_up.contiguous(), down.contiguous()
                owner = _Experts()
                executor = owner._moonep_executor
                executor.read_weights = lambda: (gate_up, down)
                lifecycle = configure_moonep(owner)
                lifecycle._ensure_slots(gate_up, down)
                lifecycle.slots[0].gate_up.fill_(float('nan'))
                lifecycle.slots[0].down.fill_(float('nan'))
                plan = SimpleNamespace(group_counts=(1, 2, 3), home_count=2,
                                       weight_send=(2, 0) if sending else (0, 0), weight_recv=(0, 1))
                key = lifecycle.new_key(executor)
                lifecycle.executions[key] = executor, plan
                remote_gate_up, remote_down = gate_up[:1].clone(), down[:1].clone()

                def exchange(_executor, _plan, source_gate_up, source_down):
                    if sending and transposed:
                        self.assertTrue(source_gate_up.is_contiguous())
                        self.assertTrue(source_down.is_contiguous())
                        rows = torch.tensor([1, 0, 1])
                        actual = moonep_runtime.pack_expert_rows(source_gate_up, source_down, rows)
                        expected = torch.cat((gate_up[rows].flatten(1), down[rows].flatten(1)), 1)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    else:
                        self.assertIs(source_gate_up, gate_up)
                        if transposed:
                            torch.testing.assert_close(lifecycle.slots[0].gate_up[:2], gate_up, rtol=0, atol=0)
                        else:
                            self.assertTrue(lifecycle.slots[0].gate_up[:2].isnan().all())
                    return remote_gate_up, remote_down, []

                with mock.patch.object(lifecycle, '_exchange_weights', side_effect=exchange) as submit:
                    lease = lifecycle.prepare(key)
                    self.assertIs(lifecycle.prepare(key), lease)
                submit.assert_called_once()
                torch.testing.assert_close(lease.gate_up, torch.cat((gate_up, remote_gate_up)), rtol=0, atol=0)
                torch.testing.assert_close(lease.down, torch.cat((down, remote_down)), rtol=0, atol=0)

    @staticmethod
    def reference_candidate(totals, ranks, fill):
        """直接用 Python 供需匹配和逐专家填充；例如 surplus=3、deficit=5 两规则迁移3/5。"""
        values = totals.tolist()
        home_count = len(values) // ranks
        target_load = [sum(values) // ranks + int(rank < sum(values) % ranks) for rank in range(ranks)]
        balance = [sum(values[rank * home_count:(rank + 1) * home_count]) - target_load[rank]
                   for rank in range(ranks)]
        quotas = [[0] * ranks for _ in range(ranks)]
        for _ in range(ranks - 1):
            home = max(range(ranks), key=lambda rank: balance[rank])
            target = min(range(ranks), key=lambda rank: balance[rank])
            amount = (-balance[target] if fill else min(balance[home], -balance[target])) if balance[home] > 0 else 0
            quotas[home][target] += amount
            balance[home] -= amount
            balance[target] += amount
        result = [[0] * ranks for _ in values]
        remaining = list(values)
        for home in range(ranks):
            experts = list(range(home * home_count, (home + 1) * home_count))
            experts.sort(key=lambda expert: -float(torch.tensor(values[expert], dtype=torch.float32)))
            targets = sorted(range(ranks),
                             key=lambda rank: -float(torch.tensor(quotas[home][rank], dtype=torch.float32)))
            for target in targets:
                need = quotas[home][target]
                for expert in experts:
                    amount = min(remaining[expert], need)
                    result[expert][target] += amount
                    remaining[expert] -= amount
                    need -= amount
            for expert in experts:
                result[expert][home] += remaining[expert]
        return torch.tensor(result, dtype=torch.int64)


    def test_shared_executor_guard_resets_between_completed_microbatches(self):
        """仅同时存活的重复调用需回退；例如 no_sync 微批1反向完，微批2仍可异步。"""
        model = _Experts()
        runtime = configure_moonep(model)
        executor = model._moonep_executor
        first = runtime.new_key(executor)
        runtime.executions[first] = executor, None
        second = runtime.new_key(executor)
        runtime.executions[second] = executor, None
        self.assertIn(executor.layer, runtime.shared_executors)
        runtime.complete(second)
        self.assertIn(executor.layer, runtime.shared_executors)
        runtime.complete(first)
        self.assertFalse(runtime.shared_executors)
        third = runtime.new_key(executor)
        runtime.executions[third] = executor, None
        self.assertNotIn(executor.layer, runtime.shared_executors)
        runtime.complete(third)
        runtime.finish_step()

    def test_pack_preserves_transposes_duplicates_and_empty_rows(self):
        """检查两矩阵打包的完整字节顺序；例如专家2可重复发送，DeepSeek转置也保留布局。"""
        for dtype, transposed in itertools.product((torch.float32, torch.bfloat16), (False, True)):
            gate_up = torch.arange(4 * 3 * 6, dtype=dtype).reshape(4, 3, 6)
            down = torch.arange(4 * 3 * 3, dtype=dtype).reshape(4, 3, 3)
            if transposed:
                gate_up, down = gate_up.transpose(1, 2), down.transpose(1, 2)
            for ids in ([2, 0, 2], [], [1], [3, 2, 1, 0], [2, 0, 2, 0, 3]):
                rows = torch.tensor(ids, dtype=torch.long)
                expected = torch.cat((gate_up.index_select(0, rows).reshape(len(ids), 18),
                                      down.index_select(0, rows).reshape(len(ids), 9)), dim=1)
                actual = moonep_runtime.pack_expert_rows(gate_up, down, rows)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(actual.is_contiguous())

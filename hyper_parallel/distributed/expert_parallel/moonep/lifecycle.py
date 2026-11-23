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
"""Execution identity and retained plans shared by all MoonEP backends."""

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

import torch

from hyper_parallel.distributed.expert_parallel.moonep.checkpoint import bind_execution


@dataclass(frozen=True)
class ExecutionKey:
    """Identify one original invocation and its replay at a parameter version."""

    layer: int
    invocation: int
    version: int


class MoonEPLifecycle(ABC):
    """Own invocation identity and ordered preparation shared by every backend."""

    def __init__(self, executors: list) -> None:
        """Bind expert owners and initialize their shared retained-execution state.

        Args:
            executors: Expert backends in the model's registered execution order.
        """
        if not executors:
            raise ValueError("MoonEP requires a nonempty expert executor list")
        self.executors = executors
        self.version = self.sequence = 0
        self.executions: dict[ExecutionKey, tuple[Any, Any]] = {}
        self.trace: list[ExecutionKey] = []
        self.weight_leases: dict[ExecutionKey, Any] = {}
        self.forward_inputs: dict[int, tuple[ExecutionKey, tuple]] = {}
        self.fsdp_layers: set[int] = set()
        self.pending_weight_preparations: dict[ExecutionKey, Any] = {}
        self.backward_weight_ready: Optional[tuple[ExecutionKey, Any]] = None
        self.statistics = Counter()
        for layer, executor in enumerate(executors):
            executor.runtime, executor.layer = self, layer

    @abstractmethod
    def prepare(self, key: ExecutionKey, *, ready_events: tuple = ()) -> Any:
        """Materialize a lease through the concrete backend's storage protocol.

        Args:
            key: Invocation whose plan is already retained by this runtime.
            ready_events: Home restoration and previous EP fences for the producer stream.

        Returns:
            The backend's weight lease, reusable by its strict consumer.
        """
        raise NotImplementedError

    @staticmethod
    def _record_weight_dependency(device: torch.device) -> Any:
        """Capture existing device work without synchronizing the host."""
        if device.type == "cpu":
            return None
        backend = torch.get_device_module(device)
        event = backend.Event()
        event.record(backend.current_stream(device))
        return event

    def prepare_restored_weights(self, key: ExecutionKey) -> None:
        """Remember restored homes and prepare only the next execution.

        Args:
            key: Original or replayed invocation restored by the existing FSDP hook.

        Note:
            A future callback must not insert a wait into the shared EP stream:
            the current expert still needs that stream to reach its exit.
        """
        if key in self.weight_leases:
            return
        if key.version != self.version or key not in self.executions or key not in self.trace:
            raise RuntimeError("restored MoonEP weights require a live execution at the current version")
        if key not in self.pending_weight_preparations:
            executor, _ = self.executions[key]
            device = executor.read_weights()[0].device
            self.pending_weight_preparations[key] = self._record_weight_dependency(device)
        self._prepare_next_weights()

    def _prepare_next_weights(self) -> None:
        """Submit at a deterministic hook or EP exit, never by event polling."""
        if not self.trace:
            return
        key = self.trace[-1]
        if key not in self.pending_weight_preparations or key in self.weight_leases:
            return
        events = [self.pending_weight_preparations[key]]
        if self.backward_weight_ready is not None and self.backward_weight_ready[0] == key:
            events.append(self.backward_weight_ready[1])
        events = tuple(event for event in events if event is not None)
        with torch.profiler.record_function(f"moonep::fsdp_ep_prepare::layer={key.layer}::key={key}"):
            if events:
                self.prepare(key, ready_events=events)
            else:
                self.prepare(key)
        del self.pending_weight_preparations[key]

    def prefetch_after_backward(self, key: ExecutionKey, device: torch.device) -> None:
        """Release the immediate successor after the current EP submitted all work.

        Args:
            key: Completed true backward invocation, never checkpoint forward replay.
            device: Device of its final input gradient.

        Note:
            Keep the exit fence even without an early home callback: depth zero
            or one can deliver the successor's normal restoration notification later.
        """
        if not self.fsdp_layers:
            return
        if key in self.executions or key in self.trace:
            raise RuntimeError("MoonEP replica prefetch must follow completion of the current EP backward")
        if not self.trace:
            if self.pending_weight_preparations:
                raise RuntimeError("MoonEP backward ended with unconsumed home restoration notifications")
            self.backward_weight_ready = None
            return
        with torch.profiler.record_function(f"moonep::ep_backward_exit::layer={key.layer}::key={key}"):
            self.backward_weight_ready = self.trace[-1], self._record_weight_dependency(device)
            self._prepare_next_weights()

    def new_key(self, executor: Any) -> ExecutionKey:
        """Allocate an original invocation; checkpoint replay reuses this key.

        Args:
            executor: Registered expert owner about to execute its original forward.

        Returns:
            A key retained in original forward order until its backward completes.
        """
        self.sequence += 1
        key = ExecutionKey(executor.layer, self.sequence, self.version)
        self.trace.append(key)
        return key

    def complete(self, key: ExecutionKey) -> tuple[Any, Any]:
        """Remove one completed plan in reverse original-forward order.

        Args:
            key: Invocation whose backend has already joined its outstanding work.

        Returns:
            The retired executor and plan for backend-specific metadata cleanup.
        """
        if not self.trace or self.trace[-1] != key:
            raise RuntimeError("MoonEP backward completion must follow reverse original-forward order")
        self.trace.pop()
        return self.executions.pop(key)

    def forward_key(self, executor: Any, hidden: Any, ids: Any, scores: Any,
                    expert_weights: Any = None) -> ExecutionKey:
        """Consume the FSDP hook's invocation, or prepare an explicit standalone call.

        Args:
            executor: Expert backend at its module's actual computation boundary.
            hidden: Source token activations.
            ids: Top-K expert IDs.
            scores: Differentiable Top-K routing weights.
            expert_weights: Optional current home views supplied by a standalone adapter.

        Returns:
            The already-prepared invocation key. FSDP execution never falls back
            to replica materialization from the consumer.
        """
        pending = self.forward_inputs.pop(executor.layer, None)
        if pending is not None:
            return pending[0]
        if executor.layer in self.fsdp_layers:
            raise RuntimeError("FSDP MoonEP inputs must pass through the expert module's preparation hook")
        key = bind_execution(executor, lambda: self.new_key(executor))
        executor.prepare_forward(key, hidden, ids, scores, expert_weights=expert_weights)
        self.prepare(key)
        return key

    def require_weight_lease(self, key: ExecutionKey) -> Any:
        """Return a prepared lease without launching weight communication.

        Args:
            key: Current original or replayed invocation.

        Returns:
            The lease prepared by FSDP hooks or an explicit standalone entry.
        """
        lease = self.weight_leases.get(key)
        if lease is None:
            raise RuntimeError("MoonEP weight lease is missing; restoration hooks must prepare it before consumption")
        return lease

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, List

from PIL import Image

from .config import settings
from .metrics import metrics
from .model_runner import InferenceEngine, InferenceOutput

@dataclass
class RequestItem:
    image: Image.Image
    future: asyncio.Future

class MicroBatcher:
    def __init__(self, engine: InferenceEngine) -> None:
        self.engine = engine
        self.queue: asyncio.Queue[RequestItem] = asyncio.Queue(maxsize=settings.queue_capacity)
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()
        self._gpu_batch8_count = 0
        self._gpu_batch8_latency_sum_ms = 0.0
        self._gpu_batch8_latency_max_ms = 0.0



    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        self._stopped.set()
        if self._task:
            await self._task

    async def submit(self, img: Image.Image) -> InferenceOutput:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        item = RequestItem(image=img, future=fut)

        if self.queue.full():
            metrics.inc("queue_full_total", 1)
            raise RuntimeError("busy")

        await self.queue.put(item)
        metrics.set_gauge("queue_size", float(self.queue.qsize()))
        return await fut

    async def _worker(self) -> None:
        while not self._stopped.is_set():
            try:
                first = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            batch: List[RequestItem] = [first]
            start = time.perf_counter()

            while len(batch) < settings.max_batch_size:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                remaining = max(settings.max_wait_ms - elapsed_ms, 0.0) / 1000.0
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch.append(nxt)
                except asyncio.TimeoutError:
                    break

            metrics.inc("batches_total", 1)
            metrics.set_gauge("last_batch_size", float(len(batch)))
            metrics.set_gauge("queue_size", float(self.queue.qsize()))

            images = [x.image for x in batch]
            try:
                out = await asyncio.to_thread(self.engine.infer, images)

                # Record GPU-only latency for full microbatches of 8.
                if len(batch)>= 4 and len(batch)>= 8:
                    self._gpu_batch8_count += 1
                    self._gpu_batch8_latency_sum_ms += float(out.latency_ms)
                    self._gpu_batch8_latency_max_ms = max(
                        self._gpu_batch8_latency_max_ms, float(out.latency_ms)
                    )

                    metrics.inc("gpu_batch8_total", 1)
                    metrics.set_gauge("gpu_batch8_latency_ms_last", float(out.latency_ms))
                    metrics.set_gauge(
                        "gpu_batch8_latency_ms_avg",
                        self._gpu_batch8_latency_sum_ms / self._gpu_batch8_count,
                    )
                    metrics.set_gauge("gpu_batch8_latency_ms_max", self._gpu_batch8_latency_max_ms)

                for i, req in enumerate(batch):
                    if not req.future.done():
                        per = InferenceOutput(
                            detections=[out.detections[i]],
                            backend=out.backend,
                            latency_ms=out.latency_ms,
                        )
                        req.future.set_result(per)
            except Exception as e:
                metrics.inc("batch_failures_total", 1)
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)
            finally:
                for _ in batch:
                    self.queue.task_done()

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from continuous_batching.config import InferenceConfig
from continuous_batching.engine import ModelEngine
from continuous_batching.models import GenerationRequest, SequenceState
from continuous_batching.scheduler import BatchScheduler, QueueFullError


class GenerateBody(BaseModel):
    prompt: str = Field(min_length=1)
    request_id: str | None = None
    max_new_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = 0


def _result(state: SequenceState, engine: ModelEngine) -> dict[str, Any]:
    return {
        "request_id": state.request_id,
        "text": engine.decode(state.generated_token_ids),
        "token_ids": state.generated_token_ids,
        "output_token_count": state.output_tokens,
    }


class InferenceService:
    def __init__(
        self,
        scheduler: BatchScheduler,
    ) -> None:
        self.scheduler = scheduler
        self._task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._futures: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._streams: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._fatal_error: Exception | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="continuous-batching")

    async def stop(self) -> None:
        if self._task is not None:
            if self._task.done():
                self._task.exception()
            else:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
            self._task = None

    async def submit(
        self, body: GenerateBody
    ) -> tuple[str, asyncio.Future[dict[str, Any]], asyncio.Queue[dict[str, Any]]]:
        if self._fatal_error is not None:
            raise RuntimeError("The inference worker is unavailable") from self._fatal_error
        request_data = body.model_dump()
        if request_data["request_id"] is None:
            request_data.pop("request_id")
        request = GenerationRequest(**request_data)
        request_id = request.request_id
        if request_id in self._futures:
            raise ValueError(f"Duplicate request_id {request_id!r}")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        stream: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._futures[request_id] = future
        self._streams[request_id] = stream
        try:
            await asyncio.to_thread(self.scheduler.submit, request)
        except Exception:
            self._futures.pop(request_id, None)
            self._streams.pop(request_id, None)
            raise
        self._wake.set()
        return request_id, future, stream

    async def _run(self) -> None:
        while True:
            if not self.scheduler.has_work:
                self._wake.clear()
                await self._wake.wait()
            try:
                await asyncio.to_thread(self.scheduler.step)
            except Exception as exc:
                self._fatal_error = exc
                await asyncio.to_thread(self.scheduler.fail_all, exc)
                for future in self._futures.values():
                    if not future.done():
                        future.set_exception(exc)
                for request_id, stream in list(self._streams.items()):
                    await stream.put(
                        {
                            "request_id": request_id,
                            "error": str(exc),
                            "finished": True,
                        }
                    )
                self._futures.clear()
                self._streams.clear()
                raise
            for event in self.scheduler.drain_token_events():
                stream = self._streams.get(event.request_id)
                if stream is not None:
                    await stream.put(
                        {
                            "request_id": event.request_id,
                            "token_id": event.token_id,
                            "text": event.text,
                            "delta": event.delta,
                            "finished": event.finished,
                        }
                    )
            for state in self.scheduler.drain_completed():
                result = _result(state, self.scheduler.engine)
                future = self._futures.pop(state.request_id, None)
                if future is not None and not future.done():
                    future.set_result(result)
                self._streams.pop(state.request_id, None)
            await asyncio.sleep(0)


def create_app(
    config: InferenceConfig,
    *,
    engine: ModelEngine | None = None,
    queue_capacity: int = 1024,
    metrics_path: str | Path | None = None,
    slo_ttft_ms: float = 500.0,
    slo_end_to_end_ms: float = 5_000.0,
) -> FastAPI:
    from continuous_batching.metrics import MetricsCollector

    model_engine = engine or ModelEngine(config)
    scheduler = BatchScheduler(
        config,
        model_engine,
        max_queue_size=queue_capacity,
        metrics=MetricsCollector(
            config,
            jsonl_path=metrics_path,
            slo_ttft_ms=slo_ttft_ms,
            slo_end_to_end_ms=slo_end_to_end_ms,
            model_metadata=model_engine.metadata(),
        ),
    )
    service = InferenceService(scheduler)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await service.start()
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(title="Continuous Batching Inference", lifespan=lifespan)
    app.state.service = service

    async def submit_or_error(
        body: GenerateBody,
    ) -> tuple[str, asyncio.Future[dict[str, Any]], asyncio.Queue[dict[str, Any]]]:
        try:
            return await service.submit(body)
        except QueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/generate_sync")
    async def generate_sync(body: GenerateBody) -> dict[str, Any]:
        _, future, _ = await submit_or_error(body)
        return await future

    @app.post("/generate")
    async def generate(body: GenerateBody) -> StreamingResponse:
        request_id, _, stream = await submit_or_error(body)

        async def events() -> AsyncIterator[str]:
            while True:
                event = await stream.get()
                yield f"data: {json.dumps(event)}\n\n"
                if event["finished"]:
                    break

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"X-Request-ID": request_id},
        )

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return scheduler.metrics.snapshot()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the continuous batching server")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mode", choices=("sequential", "static", "continuous"), default="continuous"
    )
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--static-batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--kv-cache-budget-tokens", type=int)
    parser.add_argument("--default-max-new-tokens", type=int, default=128)
    parser.add_argument("--queue-capacity", type=int, default=1024)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--slo-ttft-ms", type=float, default=500.0)
    parser.add_argument("--slo-end-to-end-ms", type=float, default=5_000.0)
    args = parser.parse_args()

    import uvicorn

    config = InferenceConfig(
        model_name=args.model_name,
        dtype=args.dtype,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        kv_cache_budget_tokens=args.kv_cache_budget_tokens,
        default_max_new_tokens=args.default_max_new_tokens,
        device=args.device,
        mode=args.mode,
        static_batch_size=args.static_batch_size,
    )
    uvicorn.run(
        create_app(
            config,
            queue_capacity=args.queue_capacity,
            metrics_path=args.metrics_path,
            slo_ttft_ms=args.slo_ttft_ms,
            slo_end_to_end_ms=args.slo_end_to_end_ms,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()

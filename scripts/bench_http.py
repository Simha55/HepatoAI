import argparse
import asyncio
import csv
import json
import math
import os
import random
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx


TERMINAL_ASSET_STATES = {"done", "done_with_failures"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((p / 100.0) * (len(s) - 1)))
    k = max(0, min(k, len(s) - 1))
    return float(s[k])


def summarize_latencies(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    return {
        "min": float(min(latencies_ms)),
        "max": float(max(latencies_ms)),
        "mean": float(statistics.mean(latencies_ms)),
        "std": float(statistics.pstdev(latencies_ms)) if len(latencies_ms) > 1 else 0.0,
        "p50": percentile(latencies_ms, 50),
        "p90": percentile(latencies_ms, 90),
        "p95": percentile(latencies_ms, 95),
        "p99": percentile(latencies_ms, 99),
    }


def parse_concurrency_list(raw: str) -> List[int]:
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Concurrency must be > 0, got {value}")
        out.append(value)
    if not out:
        raise ValueError("No valid concurrency values provided")
    return out


def parse_prometheus_text(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        name, val = parts
        try:
            metrics[name] = float(val)
        except ValueError:
            continue
    return metrics


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"images-dir not found or not a directory: {images_dir}")

    images = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    images.sort()
    if not images:
        raise RuntimeError(f"No image files found under: {images_dir}")
    return images


def mime_for(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def classify_backend(backend: Optional[str], prefer_gpu_hint: bool) -> Literal["gpu", "cpu", "unknown"]:
    if not backend:
        return "unknown"

    b = str(backend).lower()
    if "cuda" in b:
        return "gpu"
    if "cpu" in b:
        return "cpu"
    if "yolo_ultralytics_onnx" in b and prefer_gpu_hint:
        return "gpu"
    return "unknown"


class FailureLog:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(
        self,
        *,
        scenario: str,
        endpoint: str,
        phase: str,
        status_code: Optional[int],
        error: Optional[str],
        latency_ms: Optional[float],
        run_id: Optional[int] = None,
        concurrency: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        row = {
            "ts": now_iso(),
            "scenario": scenario,
            "endpoint": endpoint,
            "phase": phase,
            "status_code": status_code,
            "error": error,
            "latency_ms": latency_ms,
            "run_id": run_id,
            "concurrency": concurrency,
            "details": details or {},
        }
        self.rows.append(row)


class BenchmarkHarness:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.base_url = args.base_url.rstrip("/")
        self.timeout = httpx.Timeout(args.timeout_s)
        self.rng = random.Random(args.seed)
        self.failures = FailureLog()

        self.health_rows: List[Dict[str, Any]] = []
        self.metrics_snapshots: List[Dict[str, Any]] = []
        self.detect_rows: List[Dict[str, Any]] = []
        self.assets_rows: List[Dict[str, Any]] = []

        self.phase_ref: Dict[str, str] = {"phase": "init"}
        self._health_stop = asyncio.Event()

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(args.output_dir) / f"benchmark_{timestamp}"

        image_paths = list_images(Path(args.images_dir))
        self.image_pool: List[Tuple[str, bytes, str]] = []
        for p in image_paths:
            self.image_pool.append((p.name, p.read_bytes(), mime_for(p)))

    async def run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            health_task = asyncio.create_task(self._health_sampler(client))
            try:
                self.phase_ref["phase"] = "precheck"
                await self._snapshot_metrics(client, "pre")
                await self._health_once(client, "pre")

                self.phase_ref["phase"] = "detect"
                await self._detect_sweep(client)
                await self._snapshot_metrics(client, "post_detect")

                self.phase_ref["phase"] = "assets"
                await self._assets_workflow(client)
                await self._snapshot_metrics(client, "post_assets")

                self.phase_ref["phase"] = "cooldown"
                await asyncio.sleep(0.5)
                await self._health_once(client, "post")
                await self._snapshot_metrics(client, "final")
            finally:
                self._health_stop.set()
                await health_task

        self._write_artifacts()

    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        endpoint: str,
        *,
        files: Optional[List[Tuple[str, Tuple[str, bytes, str]]]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = self.base_url + endpoint
        start = time.perf_counter()
        try:
            resp = await client.request(method, url, files=files, params=params, headers=headers)
            dt_ms = (time.perf_counter() - start) * 1000.0
            content_type = resp.headers.get("content-type", "")
            body: Any
            if "application/json" in content_type:
                body = resp.json()
            else:
                body = resp.text
            return {
                "ok": resp.status_code == 200,
                "status_code": resp.status_code,
                "latency_ms": dt_ms,
                "body": body,
                "error": None,
            }
        except Exception as exc:
            dt_ms = (time.perf_counter() - start) * 1000.0
            return {
                "ok": False,
                "status_code": None,
                "latency_ms": dt_ms,
                "body": None,
                "error": str(exc),
            }

    async def _health_once(self, client: httpx.AsyncClient, marker: str) -> None:
        res = await self._request(client, "GET", "/health")
        body = res.get("body") if isinstance(res.get("body"), dict) else {}
        self.health_rows.append(
            {
                "ts": now_iso(),
                "phase": f"health_{marker}",
                "status_code": res["status_code"],
                "ok": int(res["ok"]),
                "latency_ms": round(res["latency_ms"], 3),
                "service_up": body.get("status") == "ok",
                "gpu_healthy": body.get("gpu_healthy"),
                "queue_size": body.get("queue_size"),
                "use_onnx": body.get("use_onnx"),
                "model_dir": body.get("model_dir"),
                "error": res["error"],
            }
        )
        if not res["ok"]:
            self.failures.add(
                scenario="health",
                endpoint="/health",
                phase=marker,
                status_code=res["status_code"],
                error=res["error"],
                latency_ms=res["latency_ms"],
            )

    async def _health_sampler(self, client: httpx.AsyncClient) -> None:
        interval_s = max(0.1, self.args.health_sample_ms / 1000.0)
        while not self._health_stop.is_set():
            phase = self.phase_ref["phase"]
            res = await self._request(client, "GET", "/health")
            body = res.get("body") if isinstance(res.get("body"), dict) else {}
            self.health_rows.append(
                {
                    "ts": now_iso(),
                    "phase": phase,
                    "status_code": res["status_code"],
                    "ok": int(res["ok"]),
                    "latency_ms": round(res["latency_ms"], 3),
                    "service_up": body.get("status") == "ok",
                    "gpu_healthy": body.get("gpu_healthy"),
                    "queue_size": body.get("queue_size"),
                    "use_onnx": body.get("use_onnx"),
                    "model_dir": body.get("model_dir"),
                    "error": res["error"],
                }
            )
            try:
                await asyncio.wait_for(self._health_stop.wait(), timeout=interval_s)
            except asyncio.TimeoutError:
                continue

    async def _snapshot_metrics(self, client: httpx.AsyncClient, label: str) -> None:
        res = await self._request(client, "GET", "/metrics")
        parsed: Dict[str, float] = {}
        if res["ok"] and isinstance(res["body"], str):
            parsed = parse_prometheus_text(res["body"])
        else:
            self.failures.add(
                scenario="metrics",
                endpoint="/metrics",
                phase=label,
                status_code=res["status_code"],
                error=res["error"],
                latency_ms=res["latency_ms"],
            )

        selected = {
            key: parsed.get(key)
            for key in [
                "requests_total",
                "responses_total",
                "request_timeout_total",
                "request_failures_total",
                "queue_full_total",
                "cpu_fallback_total",
                "service_up",
                "queue_size",
                "request_latency_ms_p50",
                "request_latency_ms_p95",
            ]
        }

        self.metrics_snapshots.append(
            {
                "ts": now_iso(),
                "label": label,
                "status_code": res["status_code"],
                "ok": res["ok"],
                "latency_ms": round(res["latency_ms"], 3),
                "selected": selected,
                "all_metrics": parsed,
                "error": res["error"],
            }
        )

    async def _detect_sweep(self, client: httpx.AsyncClient) -> None:
        levels = parse_concurrency_list(self.args.detect_concurrency_list)

        for concurrency in levels:
            await self._run_detect_level(
                client,
                concurrency=concurrency,
                total_requests=self.args.warmup_requests,
                phase="warmup",
            )
            row = await self._run_detect_level(
                client,
                concurrency=concurrency,
                total_requests=self.args.detect_requests_per_level,
                phase="measured",
            )
            self.detect_rows.append(row)

    async def _run_detect_level(
        self,
        client: httpx.AsyncClient,
        *,
        concurrency: int,
        total_requests: int,
        phase: str,
    ) -> Dict[str, Any]:
        if total_requests <= 0:
            return {
                "phase": phase,
                "concurrency": concurrency,
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "success_rate": 0.0,
                "error_rate": 0.0,
                "rps": 0.0,
                **summarize_latencies([]),
                "status_counts": {},
            }

        sample_idx = [self.rng.randrange(len(self.image_pool)) for _ in range(total_requests)]
        q: asyncio.Queue[int] = asyncio.Queue()
        for req_id in range(total_requests):
            q.put_nowait(req_id)

        latencies: List[float] = []
        status_counts: Dict[str, int] = {}
        successes = 0

        async def worker() -> None:
            nonlocal successes
            while True:
                try:
                    req_id = q.get_nowait()
                except asyncio.QueueEmpty:
                    return

                name, data, mime = self.image_pool[sample_idx[req_id]]
                files = [("file", (name, data, mime))]
                res = await self._request(client, "POST", "/detect", files=files)

                latencies.append(res["latency_ms"])
                key = str(res["status_code"])
                status_counts[key] = status_counts.get(key, 0) + 1

                if res["ok"]:
                    successes += 1
                else:
                    self.failures.add(
                        scenario="detect",
                        endpoint="/detect",
                        phase=phase,
                        status_code=res["status_code"],
                        error=res["error"],
                        latency_ms=res["latency_ms"],
                        concurrency=concurrency,
                        details={"req_id": req_id},
                    )
                q.task_done()

        started = time.perf_counter()
        tasks = [asyncio.create_task(worker()) for _ in range(min(concurrency, total_requests))]
        await asyncio.gather(*tasks)
        elapsed_s = max(1e-9, time.perf_counter() - started)

        failed = total_requests - successes
        stats = summarize_latencies(latencies)
        row = {
            "phase": phase,
            "concurrency": concurrency,
            "total_requests": total_requests,
            "successful_requests": successes,
            "failed_requests": failed,
            "success_rate": successes / total_requests,
            "error_rate": failed / total_requests,
            "rps": total_requests / elapsed_s,
            **stats,
            "status_counts": status_counts,
        }
        return row

    async def _assets_workflow(self, client: httpx.AsyncClient) -> None:
        for run_id in range(self.args.assets_runs):
            row = await self._run_one_asset(client, run_id)
            self.assets_rows.append(row)

    def _sample_asset_files(self, run_id: int) -> List[Tuple[str, Tuple[str, bytes, str]]]:
        k = max(1, self.args.assets_batch_size)
        files: List[Tuple[str, Tuple[str, bytes, str]]] = []
        for _ in range(k):
            name, data, mime = self.image_pool[self.rng.randrange(len(self.image_pool))]
            files.append(("files", (name, data, mime)))

        if run_id == 0 and self.args.include_invalid_asset_file:
            files.append(("files", ("invalid.txt", b"not_an_image", "text/plain")))
        return files

    async def _run_one_asset(self, client: httpx.AsyncClient, run_id: int) -> Dict[str, Any]:
        files = self._sample_asset_files(run_id)

        create_started = time.perf_counter()
        create_res = await self._request(client, "POST", "/v1/assets", files=files)
        create_elapsed_ms = (time.perf_counter() - create_started) * 1000.0

        row: Dict[str, Any] = {
            "run_id": run_id,
            "create_status_code": create_res["status_code"],
            "create_ok": int(create_res["ok"]),
            "create_latency_ms": round(create_res["latency_ms"], 3),
            "create_total_elapsed_ms": round(create_elapsed_ms, 3),
            "asset_id": None,
            "asset_terminal_status": None,
            "time_to_completion_ms": None,
            "expected_images": None,
            "processed_images": None,
            "failed_images": None,
            "image_success_ratio": None,
            "gpu_image_count": None,
            "cpu_image_count": None,
            "unknown_image_count": None,
            "gpu_processing_latency_ms_sum": None,
            "cpu_processing_latency_ms_sum": None,
            "unknown_processing_latency_ms_sum": None,
            "gpu_processing_latency_ms_avg": None,
            "cpu_processing_latency_ms_avg": None,
            "gpu_processing_ratio": None,
            "cpu_processing_ratio": None,
            "report_status_code": None,
            "report_ok": 0,
            "report_latency_ms": None,
            "export_json_status_code": None,
            "export_json_ok": 0,
            "export_json_latency_ms": None,
            "export_csv_status_code": None,
            "export_csv_ok": 0,
            "export_csv_latency_ms": None,
            "poll_attempts": 0,
            "poll_errors": 0,
            "timed_out": 0,
        }

        if not create_res["ok"] or not isinstance(create_res.get("body"), dict):
            self.failures.add(
                scenario="assets",
                endpoint="/v1/assets",
                phase="create",
                status_code=create_res["status_code"],
                error=create_res["error"],
                latency_ms=create_res["latency_ms"],
                run_id=run_id,
            )
            return row

        asset_id = create_res["body"].get("asset_id")
        row["asset_id"] = asset_id
        if not asset_id:
            self.failures.add(
                scenario="assets",
                endpoint="/v1/assets",
                phase="create_parse",
                status_code=create_res["status_code"],
                error="missing asset_id",
                latency_ms=create_res["latency_ms"],
                run_id=run_id,
            )
            return row

        deadline = time.perf_counter() + self.args.asset_completion_timeout_s
        last_asset_body: Optional[Dict[str, Any]] = None

        while time.perf_counter() < deadline:
            row["poll_attempts"] += 1
            res = await self._request(client, "GET", f"/v1/assets/{asset_id}")

            if not res["ok"] or not isinstance(res.get("body"), dict):
                row["poll_errors"] += 1
                self.failures.add(
                    scenario="assets",
                    endpoint="/v1/assets/{asset_id}",
                    phase="poll",
                    status_code=res["status_code"],
                    error=res["error"],
                    latency_ms=res["latency_ms"],
                    run_id=run_id,
                    details={"asset_id": asset_id},
                )
                await asyncio.sleep(self.args.poll_interval_ms / 1000.0)
                continue

            body = res["body"]
            last_asset_body = body
            status = body.get("status")
            if status in TERMINAL_ASSET_STATES:
                row["asset_terminal_status"] = status
                break

            await asyncio.sleep(self.args.poll_interval_ms / 1000.0)

        if row["asset_terminal_status"] is None:
            row["timed_out"] = 1
            self.failures.add(
                scenario="assets",
                endpoint="/v1/assets/{asset_id}",
                phase="timeout",
                status_code=None,
                error="asset_completion_timeout",
                latency_ms=None,
                run_id=run_id,
                details={"asset_id": asset_id, "timeout_s": self.args.asset_completion_timeout_s},
            )
            return row

        row["time_to_completion_ms"] = round((time.perf_counter() - create_started) * 1000.0, 3)
        if isinstance(last_asset_body, dict):
            exp = last_asset_body.get("expected_images")
            proc = last_asset_body.get("processed_images")
            fail = last_asset_body.get("failed_images")
            row["expected_images"] = exp
            row["processed_images"] = proc
            row["failed_images"] = fail
            if isinstance(exp, int) and exp > 0 and isinstance(proc, int):
                row["image_success_ratio"] = proc / exp

            images = last_asset_body.get("images")
            if not isinstance(images, list):
                self.failures.add(
                    scenario="assets",
                    endpoint="/v1/assets/{asset_id}",
                    phase="backend_classification",
                    status_code=None,
                    error="images field missing or not list",
                    latency_ms=None,
                    run_id=run_id,
                    details={"asset_id": asset_id},
                )
                images = []

            prefer_gpu_env = os.getenv("PREFER_GPU", "1") == "1"
            prefer_gpu_hint = bool(self.args.gpu_backend_hint_from_prefer_gpu and prefer_gpu_env)

            gpu_count = 0
            cpu_count = 0
            unknown_count = 0
            gpu_sum = 0.0
            cpu_sum = 0.0
            unknown_sum = 0.0

            for idx, im in enumerate(images):
                if not isinstance(im, dict):
                    self.failures.add(
                        scenario="assets",
                        endpoint="/v1/assets/{asset_id}",
                        phase="backend_classification",
                        status_code=None,
                        error="image row not object",
                        latency_ms=None,
                        run_id=run_id,
                        details={"asset_id": asset_id, "image_index": idx},
                    )
                    continue

                backend = im.get("backend")
                if backend is None:
                    self.failures.add(
                        scenario="assets",
                        endpoint="/v1/assets/{asset_id}",
                        phase="backend_classification",
                        status_code=None,
                        error="missing backend",
                        latency_ms=None,
                        run_id=run_id,
                        details={"asset_id": asset_id, "image_id": im.get("image_id")},
                    )
                category = classify_backend(backend, prefer_gpu_hint=prefer_gpu_hint)

                lat_val = im.get("latency_ms")
                latency_ok = False
                latency_ms = 0.0
                if isinstance(lat_val, (int, float)):
                    latency_ok = True
                    latency_ms = float(lat_val)
                elif lat_val is not None:
                    self.failures.add(
                        scenario="assets",
                        endpoint="/v1/assets/{asset_id}",
                        phase="backend_classification",
                        status_code=None,
                        error="non-numeric latency_ms",
                        latency_ms=None,
                        run_id=run_id,
                        details={"asset_id": asset_id, "image_id": im.get("image_id"), "latency_ms": lat_val},
                    )
                else:
                    self.failures.add(
                        scenario="assets",
                        endpoint="/v1/assets/{asset_id}",
                        phase="backend_classification",
                        status_code=None,
                        error="missing latency_ms",
                        latency_ms=None,
                        run_id=run_id,
                        details={"asset_id": asset_id, "image_id": im.get("image_id")},
                    )

                if category == "gpu":
                    gpu_count += 1
                    if latency_ok:
                        gpu_sum += latency_ms
                elif category == "cpu":
                    cpu_count += 1
                    if latency_ok:
                        cpu_sum += latency_ms
                else:
                    unknown_count += 1
                    if latency_ok:
                        unknown_sum += latency_ms

            total_images = gpu_count + cpu_count + unknown_count
            if total_images == 0:
                self.failures.add(
                    scenario="assets",
                    endpoint="/v1/assets/{asset_id}",
                    phase="backend_classification",
                    status_code=None,
                    error="asset has zero images rows",
                    latency_ms=None,
                    run_id=run_id,
                    details={"asset_id": asset_id},
                )

            row["gpu_image_count"] = gpu_count
            row["cpu_image_count"] = cpu_count
            row["unknown_image_count"] = unknown_count
            row["gpu_processing_latency_ms_sum"] = gpu_sum
            row["cpu_processing_latency_ms_sum"] = cpu_sum
            row["unknown_processing_latency_ms_sum"] = unknown_sum
            row["gpu_processing_latency_ms_avg"] = (gpu_sum / gpu_count) if gpu_count > 0 else 0.0
            row["cpu_processing_latency_ms_avg"] = (cpu_sum / cpu_count) if cpu_count > 0 else 0.0
            row["gpu_processing_ratio"] = (gpu_count / total_images) if total_images > 0 else None
            row["cpu_processing_ratio"] = (cpu_count / total_images) if total_images > 0 else None

        report_res = await self._request(client, "GET", f"/v1/assets/{asset_id}/report")
        row["report_status_code"] = report_res["status_code"]
        row["report_ok"] = int(report_res["ok"])
        row["report_latency_ms"] = round(report_res["latency_ms"], 3)
        if not report_res["ok"]:
            self.failures.add(
                scenario="assets",
                endpoint="/v1/assets/{asset_id}/report",
                phase="report",
                status_code=report_res["status_code"],
                error=report_res["error"],
                latency_ms=report_res["latency_ms"],
                run_id=run_id,
                details={"asset_id": asset_id},
            )

        exp_json_res = await self._request(
            client,
            "GET",
            f"/v1/assets/{asset_id}/export",
            params={"format": "json"},
        )
        row["export_json_status_code"] = exp_json_res["status_code"]
        row["export_json_ok"] = int(exp_json_res["ok"])
        row["export_json_latency_ms"] = round(exp_json_res["latency_ms"], 3)
        if not exp_json_res["ok"]:
            self.failures.add(
                scenario="assets",
                endpoint="/v1/assets/{asset_id}/export?format=json",
                phase="export_json",
                status_code=exp_json_res["status_code"],
                error=exp_json_res["error"],
                latency_ms=exp_json_res["latency_ms"],
                run_id=run_id,
                details={"asset_id": asset_id},
            )

        exp_csv_res = await self._request(
            client,
            "GET",
            f"/v1/assets/{asset_id}/export",
            params={"format": "csv"},
        )
        row["export_csv_status_code"] = exp_csv_res["status_code"]
        row["export_csv_ok"] = int(exp_csv_res["ok"])
        row["export_csv_latency_ms"] = round(exp_csv_res["latency_ms"], 3)
        if not exp_csv_res["ok"]:
            self.failures.add(
                scenario="assets",
                endpoint="/v1/assets/{asset_id}/export?format=csv",
                phase="export_csv",
                status_code=exp_csv_res["status_code"],
                error=exp_csv_res["error"],
                latency_ms=exp_csv_res["latency_ms"],
                run_id=run_id,
                details={"asset_id": asset_id},
            )

        return row

    def _write_csv(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return

        fieldnames: List[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_failures_jsonl(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in self.failures.rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _build_summary(self) -> Dict[str, Any]:
        detect_measured = [r for r in self.detect_rows if r.get("phase") == "measured"]
        detect_sorted_p95 = sorted(detect_measured, key=lambda r: r.get("p95", math.inf))
        detect_sorted_rps = sorted(detect_measured, key=lambda r: r.get("rps", -math.inf), reverse=True)

        best_conc = detect_sorted_p95[0] if detect_sorted_p95 else None
        worst_conc = detect_sorted_p95[-1] if detect_sorted_p95 else None
        median_conc = detect_sorted_p95[len(detect_sorted_p95) // 2] if detect_sorted_p95 else None

        asset_completion = [r for r in self.assets_rows if r.get("time_to_completion_ms") is not None]
        ttc_values = [float(r["time_to_completion_ms"]) for r in asset_completion]

        report_lats = [float(r["report_latency_ms"]) for r in self.assets_rows if r.get("report_latency_ms") is not None]
        exp_json_lats = [float(r["export_json_latency_ms"]) for r in self.assets_rows if r.get("export_json_latency_ms") is not None]
        exp_csv_lats = [float(r["export_csv_latency_ms"]) for r in self.assets_rows if r.get("export_csv_latency_ms") is not None]
        gpu_sum_lats = [
            float(r["gpu_processing_latency_ms_sum"])
            for r in self.assets_rows
            if r.get("gpu_processing_latency_ms_sum") is not None
        ]
        cpu_sum_lats = [
            float(r["cpu_processing_latency_ms_sum"])
            for r in self.assets_rows
            if r.get("cpu_processing_latency_ms_sum") is not None
        ]
        gpu_ratio_vals = [
            float(r["gpu_processing_ratio"])
            for r in self.assets_rows
            if r.get("gpu_processing_ratio") is not None
        ]

        total_detect_requests = sum(int(r.get("total_requests", 0)) for r in detect_measured)
        total_detect_success = sum(int(r.get("successful_requests", 0)) for r in detect_measured)

        total_assets_runs = len(self.assets_rows)
        assets_completed = sum(1 for r in self.assets_rows if r.get("asset_terminal_status") in TERMINAL_ASSET_STATES)
        assets_timed_out = sum(1 for r in self.assets_rows if r.get("timed_out") == 1)
        backend_split_totals = {
            "gpu_image_count": int(
                sum(int(r.get("gpu_image_count", 0) or 0) for r in self.assets_rows)
            ),
            "cpu_image_count": int(
                sum(int(r.get("cpu_image_count", 0) or 0) for r in self.assets_rows)
            ),
            "unknown_image_count": int(
                sum(int(r.get("unknown_image_count", 0) or 0) for r in self.assets_rows)
            ),
        }
        mixed_backend_assets_count = 0
        gpu_only_assets_count = 0
        cpu_only_assets_count = 0
        unknown_backend_assets_count = 0

        for r in self.assets_rows:
            g = int(r.get("gpu_image_count", 0) or 0)
            c = int(r.get("cpu_image_count", 0) or 0)
            u = int(r.get("unknown_image_count", 0) or 0)
            kinds = int(g > 0) + int(c > 0) + int(u > 0)
            if kinds >= 2:
                mixed_backend_assets_count += 1
            if g > 0 and c == 0 and u == 0:
                gpu_only_assets_count += 1
            if c > 0 and g == 0 and u == 0:
                cpu_only_assets_count += 1
            if u > 0 and g == 0 and c == 0:
                unknown_backend_assets_count += 1

        health_ok = sum(1 for r in self.health_rows if r.get("ok") == 1)
        health_total = len(self.health_rows)

        reliability = {
            "detect_success_rate": (total_detect_success / total_detect_requests) if total_detect_requests else 0.0,
            "assets_completion_rate": (assets_completed / total_assets_runs) if total_assets_runs else 0.0,
            "assets_timeout_rate": (assets_timed_out / total_assets_runs) if total_assets_runs else 0.0,
            "health_ok_rate": (health_ok / health_total) if health_total else 0.0,
            "failure_events": len(self.failures.rows),
        }

        return {
            "run_metadata": {
                "generated_at": now_iso(),
                "base_url": self.base_url,
                "images_dir": str(self.args.images_dir),
                "output_dir": str(self.run_dir),
                "seed": self.args.seed,
                "settings": {
                    "warmup_requests": self.args.warmup_requests,
                    "timeout_s": self.args.timeout_s,
                    "detect_concurrency_list": parse_concurrency_list(self.args.detect_concurrency_list),
                    "detect_requests_per_level": self.args.detect_requests_per_level,
                    "assets_batch_size": self.args.assets_batch_size,
                    "assets_runs": self.args.assets_runs,
                    "poll_interval_ms": self.args.poll_interval_ms,
                    "asset_completion_timeout_s": self.args.asset_completion_timeout_s,
                    "gpu_backend_hint_from_prefer_gpu": self.args.gpu_backend_hint_from_prefer_gpu,
                },
            },
            "executive_kpis": {
                "detect": {
                    "total_requests": total_detect_requests,
                    "success_requests": total_detect_success,
                    "best_p95_concurrency": best_conc,
                    "median_p95_concurrency": median_conc,
                    "worst_p95_concurrency": worst_conc,
                    "max_rps_concurrency": detect_sorted_rps[0] if detect_sorted_rps else None,
                },
                "assets": {
                    "runs": total_assets_runs,
                    "completed": assets_completed,
                    "timed_out": assets_timed_out,
                    "ttc_stats_ms": summarize_latencies(ttc_values),
                    "report_latency_ms": summarize_latencies(report_lats),
                    "export_json_latency_ms": summarize_latencies(exp_json_lats),
                    "export_csv_latency_ms": summarize_latencies(exp_csv_lats),
                    "gpu_processing_latency_ms_sum_stats": summarize_latencies(gpu_sum_lats),
                    "cpu_processing_latency_ms_sum_stats": summarize_latencies(cpu_sum_lats),
                    "gpu_image_ratio_stats": summarize_latencies(gpu_ratio_vals),
                    "backend_split_totals": backend_split_totals,
                    "mixed_backend_assets_count": mixed_backend_assets_count,
                    "gpu_only_assets_count": gpu_only_assets_count,
                    "cpu_only_assets_count": cpu_only_assets_count,
                    "unknown_backend_assets_count": unknown_backend_assets_count,
                },
                "health": {
                    "samples": health_total,
                    "ok_samples": health_ok,
                },
            },
            "reliability_summary": reliability,
            "slide_views": {
                "inference_scalability": {
                    "title": "Inference scalability",
                    "x_axis": "concurrency",
                    "series": {
                        "p50_ms": [{"concurrency": r["concurrency"], "value": r["p50"]} for r in detect_measured],
                        "p95_ms": [{"concurrency": r["concurrency"], "value": r["p95"]} for r in detect_measured],
                        "p99_ms": [{"concurrency": r["concurrency"], "value": r["p99"]} for r in detect_measured],
                        "rps": [{"concurrency": r["concurrency"], "value": r["rps"]} for r in detect_measured],
                    },
                },
                "asset_pipeline_sla": {
                    "title": "Asset pipeline SLA",
                    "completion_rate": reliability["assets_completion_rate"],
                    "timeout_rate": reliability["assets_timeout_rate"],
                    "time_to_completion_ms": summarize_latencies(ttc_values),
                },
                "system_health_under_load": {
                    "title": "System health under load",
                    "health_ok_rate": reliability["health_ok_rate"],
                    "queue_size_samples": [
                        {
                            "ts": r["ts"],
                            "phase": r["phase"],
                            "queue_size": r.get("queue_size"),
                            "gpu_healthy": r.get("gpu_healthy"),
                        }
                        for r in self.health_rows
                    ],
                    "metrics_snapshots": self.metrics_snapshots,
                },
            },
        }

    def _write_artifacts(self) -> None:
        summary = self._build_summary()

        self._write_csv(self.run_dir / "detect_runs.csv", self.detect_rows)
        self._write_csv(self.run_dir / "assets_runs.csv", self.assets_rows)
        self._write_csv(self.run_dir / "timeseries_health.csv", self.health_rows)
        self._write_failures_jsonl(self.run_dir / "raw_failures.jsonl")

        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Rigorous API benchmark harness for AssetLens")

    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--output-dir", default="results")

    ap.add_argument("--warmup-requests", type=int, default=20)
    ap.add_argument("--timeout-s", type=float, default=30.0)

    ap.add_argument("--detect-concurrency-list", default="1,2,4,8,16,32")
    ap.add_argument("--detect-requests-per-level", type=int, default=200)

    ap.add_argument("--assets-batch-size", type=int, default=8)
    ap.add_argument("--assets-runs", type=int, default=30)
    ap.add_argument("--poll-interval-ms", type=int, default=300)
    ap.add_argument("--asset-completion-timeout-s", type=float, default=120.0)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--health-sample-ms", type=int, default=1000)
    ap.add_argument("--include-invalid-asset-file", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gpu-backend-hint-from-prefer-gpu", action=argparse.BooleanOptionalAction, default=True)

    return ap


async def async_main(args: argparse.Namespace) -> int:
    harness = BenchmarkHarness(args)
    await harness.run()

    print(f"Benchmark completed. Artifacts: {harness.run_dir}")
    print(f"  - {harness.run_dir / 'summary.json'}")
    print(f"  - {harness.run_dir / 'detect_runs.csv'}")
    print(f"  - {harness.run_dir / 'assets_runs.csv'}")
    print(f"  - {harness.run_dir / 'timeseries_health.csv'}")
    print(f"  - {harness.run_dir / 'raw_failures.jsonl'}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())


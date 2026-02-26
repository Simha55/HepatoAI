import time
import threading
from collections import defaultdict, deque

class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.latency_ms = deque(maxlen=5000)

    def inc(self, name: str, value: int = 1) -> None:
        with self._lock:
            self.counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self.gauges[name] = value

    def observe_latency_ms(self, value: float) -> None:
        with self._lock:
            self.latency_ms.append(float(value))

    def snapshot(self) -> str:
        with self._lock:
            lat = sorted(self.latency_ms)
            def pct(p: float) -> float:
                if not lat:
                    return 0.0
                idx = int(p * (len(lat) - 1))
                return float(lat[idx])

            p50 = pct(0.50)
            p95 = pct(0.95)
            now = time.time()

            lines = []
            for k, v in self.counters.items():
                lines.append(f"# TYPE {k} counter")
                lines.append(f"{k} {v}")
            for k, v in self.gauges.items():
                lines.append(f"# TYPE {k} gauge")
                lines.append(f"{k} {v}")

            lines.append("# TYPE request_latency_ms_p50 gauge")
            lines.append(f"request_latency_ms_p50 {p50}")
            lines.append("# TYPE request_latency_ms_p95 gauge")
            lines.append(f"request_latency_ms_p95 {p95}")
            lines.append("# TYPE metrics_generated_at gauge")
            lines.append(f"metrics_generated_at {now}")
            return "\n".join(lines) + "\n"

metrics = Metrics()

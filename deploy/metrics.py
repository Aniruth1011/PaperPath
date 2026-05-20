"""
System + pipeline metrics store.

Collects per-query timing/stats and periodically samples system resource usage.
All data is in-memory (optionally persisted to a JSON file on shutdown).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from statistics import mean, median, quantiles
from typing import Any, Deque, Dict, List, Optional

import psutil


# ── Per-query record ──────────────────────────────────────────────────────────

@dataclass
class QueryRecord:
    query: str
    topic_label: str
    timestamp: float                # Unix timestamp

    # Stage timings in milliseconds
    total_ms: float
    retrieval_ms: float
    canon_ms: float
    bfs_ms: float
    scoring_ms: float
    curriculum_ms: float

    # BFS stats
    nodes_visited: int
    candidates: int
    lineage_nodes: int

    # Result quality
    canonical_matched: bool
    prerequisites_returned: int
    curriculum_stages: int

    # Optional error
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


# ── System snapshot ───────────────────────────────────────────────────────────

@dataclass
class SystemSnapshot:
    timestamp: float
    cpu_percent: float
    ram_percent: float
    ram_used_mb: float
    process_rss_mb: float


# ── Metrics store ─────────────────────────────────────────────────────────────

class MetricsStore:
    """Thread-safe in-memory metrics store."""

    def __init__(self, max_history: int = 500, system_poll_interval: float = 5.0):
        self._lock = threading.Lock()
        self.queries: Deque[QueryRecord] = deque(maxlen=max_history)
        self.system: Deque[SystemSnapshot] = deque(maxlen=120)   # last 10 min at 5-s intervals
        self.start_time = time.time()
        self._proc = psutil.Process()

        # Background system polling
        self._poll_interval = system_poll_interval
        self._poll_thread = threading.Thread(
            target=self._poll_system, daemon=True, name="metrics-poller"
        )
        self._poll_thread.start()

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(self, rec: QueryRecord) -> None:
        with self._lock:
            self.queries.append(rec)

    # ── Derived metrics helpers ───────────────────────────────────────────────

    def _latencies(self, records: List[QueryRecord]) -> List[float]:
        return [r.total_ms for r in records if r.success]

    def _percentile(self, values: List[float], pct: int) -> float:
        if not values:
            return 0.0
        sorted_v = sorted(values)
        idx = max(0, int(len(sorted_v) * pct / 100) - 1)
        return sorted_v[idx]

    def _qpm(self, records: List[QueryRecord], window_s: float = 60.0) -> float:
        """Queries per minute in the last `window_s` seconds."""
        cutoff = time.time() - window_s
        recent = [r for r in records if r.timestamp >= cutoff]
        return len(recent) * (60.0 / window_s)

    # ── Public summary ────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            records = list(self.queries)
            sys_snaps = list(self.system)

        if not records:
            return {
                "total_queries": 0,
                "uptime_s": round(time.time() - self.start_time),
                "system": self._latest_system(sys_snaps),
            }

        successful = [r for r in records if r.success]
        latencies = self._latencies(successful)

        # Stage averages (successful only)
        def _avg(attr: str) -> float:
            vals = [getattr(r, attr) for r in successful]
            return round(mean(vals), 1) if vals else 0.0

        # Topic distribution
        topic_counts: Dict[str, int] = {}
        for r in records:
            topic_counts[r.topic_label] = topic_counts.get(r.topic_label, 0) + 1

        # Recent 20 for trend charts
        recent = records[-20:]
        latency_trend = [
            {"query": r.query[:30], "total_ms": r.total_ms, "ts": r.timestamp}
            for r in recent
        ]

        # Stage breakdown for recent 20
        stage_breakdown = [
            {
                "retrieval": r.retrieval_ms,
                "canon":     r.canon_ms,
                "bfs":       r.bfs_ms,
                "scoring":   r.scoring_ms,
                "curriculum": r.curriculum_ms,
            }
            for r in recent
        ]

        return {
            "total_queries":       len(records),
            "successful_queries":  len(successful),
            "error_rate":          round((len(records) - len(successful)) / len(records), 4),
            "canonical_match_rate": round(
                sum(1 for r in successful if r.canonical_matched) / max(len(successful), 1), 4
            ),
            "qpm":                 round(self._qpm(records), 2),
            "latency": {
                "p50_ms":  round(self._percentile(latencies, 50)),
                "p90_ms":  round(self._percentile(latencies, 90)),
                "p99_ms":  round(self._percentile(latencies, 99)),
                "avg_ms":  round(mean(latencies)) if latencies else 0,
                "min_ms":  round(min(latencies)) if latencies else 0,
                "max_ms":  round(max(latencies)) if latencies else 0,
            },
            "stage_avg_ms": {
                "retrieval":  _avg("retrieval_ms"),
                "canon":      _avg("canon_ms"),
                "bfs":        _avg("bfs_ms"),
                "scoring":    _avg("scoring_ms"),
                "curriculum": _avg("curriculum_ms"),
            },
            "bfs_avg": {
                "nodes_visited": round(_avg("nodes_visited")),
                "candidates":    round(_avg("candidates")),
                "lineage_nodes": round(_avg("lineage_nodes")),
            },
            "topic_distribution": topic_counts,
            "latency_trend":       latency_trend,
            "stage_breakdown":     stage_breakdown,
            "recent_queries": [
                {
                    "query":             r.query,
                    "topic":             r.topic_label,
                    "total_ms":          round(r.total_ms),
                    "canonical_matched": r.canonical_matched,
                    "prerequisites":     r.prerequisites_returned,
                    "stages":            r.curriculum_stages,
                    "error":             r.error,
                    "ts":                r.timestamp,
                }
                for r in reversed(records[-10:])
            ],
            "uptime_s": round(time.time() - self.start_time),
            "system":   self._latest_system(sys_snaps),
            "system_trend": [
                {"cpu": s.cpu_percent, "ram": s.ram_percent, "ts": s.timestamp}
                for s in sys_snaps[-30:]   # last 2.5 min
            ],
        }

    # ── System poller ─────────────────────────────────────────────────────────

    def _latest_system(self, snaps: List[SystemSnapshot]) -> Dict:
        if not snaps:
            return {}
        s = snaps[-1]
        return {
            "cpu_percent":     s.cpu_percent,
            "ram_percent":     s.ram_percent,
            "ram_used_mb":     s.ram_used_mb,
            "process_rss_mb":  s.process_rss_mb,
        }

    def _poll_system(self) -> None:
        while True:
            try:
                ram = psutil.virtual_memory()
                snap = SystemSnapshot(
                    timestamp=time.time(),
                    cpu_percent=psutil.cpu_percent(interval=None),
                    ram_percent=ram.percent,
                    ram_used_mb=round(ram.used / 1024 ** 2),
                    process_rss_mb=round(self._proc.memory_info().rss / 1024 ** 2),
                )
                with self._lock:
                    self.system.append(snap)
            except Exception:
                pass
            time.sleep(self._poll_interval)

import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.events = deque(maxlen=max_history)
        self.counters = defaultdict(int)
        self.lock = threading.Lock()

    def record_metric(self, name: str, value: float, tags: Dict[str, Any] = None):
        """Record a numeric metric"""
        with self.lock:
            self.metrics[name].append(
                {
                    "value": value,
                    "timestamp": datetime.utcnow().isoformat(),
                    "tags": tags or {},
                }
            )

    def record_event(self, event_type: str, data: Dict[str, Any] = None):
        """Record an event"""
        with self.lock:
            self.events.append(
                {
                    "type": event_type,
                    "data": data or {},
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            self.counters[event_type] += 1

    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter"""
        with self.lock:
            self.counters[name] += value

    def get_metric(self, name: str) -> Optional[List[Dict[str, Any]]]:
        """Get all values for a specific metric"""
        with self.lock:
            if name not in self.metrics:
                return None
            return list(self.metrics[name])

    def get_metric_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get statistics for a metric"""
        metric_data = self.get_metric(name)
        if not metric_data:
            return None

        values = [m["value"] for m in metric_data]
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1] if values else None,
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics and statistics"""
        with self.lock:
            result = {
                "metrics": {},
                "counters": dict(self.counters),
                "recent_events": list(self.events)[-100:],  # Last 100 events
                "timestamp": datetime.utcnow().isoformat(),
            }

            for name in self.metrics.keys():
                result["metrics"][name] = self.get_metric_stats(name)

            return result

    def get_events(
        self, event_type: str = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent events, optionally filtered by type"""
        with self.lock:
            events = list(self.events)

            if event_type:
                events = [e for e in events if e["type"] == event_type]

            return events[-limit:]

    def get_counter(self, name: str) -> int:
        """Get counter value"""
        with self.lock:
            return self.counters.get(name, 0)

    def reset_metrics(self):
        """Reset all metrics"""
        with self.lock:
            self.metrics.clear()
            self.events.clear()
            self.counters.clear()
        logger.info("All metrics reset")

    def export_metrics(self) -> Dict[str, Any]:
        """Export metrics in Prometheus-like format"""
        with self.lock:
            export_data = {"timestamp": datetime.utcnow().isoformat(), "metrics": []}

            # Export counters
            for name, value in self.counters.items():
                export_data["metrics"].append(
                    {"name": name, "type": "counter", "value": value}
                )

            # Export gauge metrics (latest value)
            for name, data in self.metrics.items():
                if data:
                    latest = data[-1]
                    export_data["metrics"].append(
                        {
                            "name": name,
                            "type": "gauge",
                            "value": latest["value"],
                            "timestamp": latest["timestamp"],
                        }
                    )

            return export_data


class PerformanceMonitor:
    """Context manager for monitoring code performance"""

    def __init__(self, metrics_collector: MetricsCollector, operation_name: str):
        self.metrics_collector = metrics_collector
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.time() - self.start_time) * 1000  # milliseconds
        self.metrics_collector.record_metric(
            f"{self.operation_name}_duration_ms",
            duration,
            tags={"success": exc_type is None},
        )

        if exc_type is not None:
            self.metrics_collector.record_event(
                f"{self.operation_name}_failed", {"error": str(exc_val)}
            )

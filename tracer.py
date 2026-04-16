"""
Structured JSON logging and tracing for every pipeline stage.

Usage:
    tracer = PipelineTracer()
    with tracer.span("research_agent", request_id) as span:
        span.set_tokens(tokens_in=2400, tokens_out=1800)
        span.add_tool_call("search_places", latency_ms=320, cache_hit=False)
    # On exit the span is serialised and emitted to the logger.
"""

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Configure the pipeline logger to emit one JSON line per record
# ---------------------------------------------------------------------------

class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # record.msg is already a dict when emitted by Span
        if isinstance(record.msg, dict):
            return json.dumps(record.msg, default=str)
        return json.dumps({"message": record.getMessage()}, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Call once at startup to wire up the JSON pipeline logger."""
    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


# ---------------------------------------------------------------------------
# Span — one log entry per pipeline stage
# ---------------------------------------------------------------------------

@dataclass
class Span:
    stage: str
    request_id: str
    _start: float = field(default_factory=time.monotonic, init=False, repr=False)
    _timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        init=False,
        repr=False,
    )
    tokens_in: int = 0
    tokens_out: int = 0
    retry_count: int = 0
    router_intent: str | None = None
    validation_result: str | None = None          # "pass" | "fail" | "partial"
    tool_calls: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)     # any ad-hoc fields

    # ── Mutation helpers ────────────────────────────────────────────────────

    def set_tokens(self, tokens_in: int, tokens_out: int) -> None:
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out

    def add_tool_call(self, name: str, latency_ms: float, cache_hit: bool) -> None:
        self.tool_calls.append({
            "name": name,
            "latency_ms": round(latency_ms),
            "cache_hit": cache_hit,
        })

    def set(self, **kwargs) -> None:
        """Set arbitrary extra fields (surfaced under the span's top-level JSON)."""
        self.extra.update(kwargs)

    # ── Serialisation ───────────────────────────────────────────────────────

    def to_dict(self, latency_ms: float) -> dict:
        entry = {
            "request_id":       self.request_id,
            "timestamp":        self._timestamp,
            "stage":            self.stage,
            "latency_ms":       round(latency_ms),
            "tokens_in":        self.tokens_in,
            "tokens_out":       self.tokens_out,
            "tool_calls":       self.tool_calls,
            "total_tool_calls": len(self.tool_calls),
            "retry_count":      self.retry_count,
            "router_intent":    self.router_intent,
            "validation_result": self.validation_result,
        }
        entry.update(self.extra)
        return entry

    def emit(self) -> dict:
        elapsed_ms = (time.monotonic() - self._start) * 1000
        entry = self.to_dict(elapsed_ms)
        logger.info(entry)
        return entry


# ---------------------------------------------------------------------------
# PipelineTracer — factory + context manager
# ---------------------------------------------------------------------------

class PipelineTracer:
    """
    Creates and tracks Span objects for a single conversation/request.
    One PipelineTracer per TripOrchestrator session.
    """

    def __init__(self, request_id: str | None = None):
        self.request_id: str = request_id or str(uuid.uuid4())
        self._spans: list[dict] = []

    def new_span(self, stage: str) -> Span:
        return Span(stage=stage, request_id=self.request_id)

    @contextmanager
    def span(self, stage: str):
        """
        Context manager that creates a Span, yields it, then emits the log
        entry on exit (even if an exception is raised).

        Example:
            with tracer.span("planning_agent") as s:
                s.set_tokens(1200, 900)
                s.retry_count = 1
                s.validation_result = "pass"
        """
        s = self.new_span(stage)
        try:
            yield s
        finally:
            entry = s.emit()
            self._spans.append(entry)

    def summary(self) -> dict:
        """Aggregate view across all spans for this request."""
        total_latency = sum(e.get("latency_ms", 0) for e in self._spans)
        total_tokens_in = sum(e.get("tokens_in", 0) for e in self._spans)
        total_tokens_out = sum(e.get("tokens_out", 0) for e in self._spans)
        total_tool_calls = sum(e.get("total_tool_calls", 0) for e in self._spans)
        cache_hits = sum(
            1 for e in self._spans
            for tc in e.get("tool_calls", [])
            if tc.get("cache_hit")
        )
        cache_total = sum(
            len(e.get("tool_calls", [])) for e in self._spans
        )
        return {
            "request_id":       self.request_id,
            "stages":           [e["stage"] for e in self._spans],
            "total_latency_ms": round(total_latency),
            "total_tokens_in":  total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_tool_calls": total_tool_calls,
            "cache_hits":       cache_hits,
            "cache_total":      cache_total,
            "cache_hit_rate":   f"{(cache_hits / cache_total * 100):.1f}%" if cache_total else "N/A",
        }

    @property
    def spans(self) -> list[dict]:
        return list(self._spans)

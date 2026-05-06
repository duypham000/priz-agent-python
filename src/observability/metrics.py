"""Prometheus metrics for pagent."""

from prometheus_client import Counter, Histogram

token_count_total = Counter(
    "pagent_token_count_total",
    "Total LLM tokens used",
    ["model", "user", "team"],
)

agent_latency_seconds = Histogram(
    "pagent_agent_latency_seconds",
    "Agent execution latency in seconds",
    ["agent", "team"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

tool_calls_total = Counter(
    "pagent_tool_calls_total",
    "Total tool invocations",
    ["tool"],
)

hitl_total = Counter(
    "pagent_hitl_total",
    "Total HITL approval requests",
    ["team"],
)

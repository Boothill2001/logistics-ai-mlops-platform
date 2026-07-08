"""Prometheus metrics.

Kept in one module so the metric surface is reviewable at a glance —
in production this is effectively your alerting API.
"""
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "status"]
)
ERROR_COUNT = Counter(
    "api_errors_total", "Total API errors", ["endpoint", "error_type"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency", ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
# Distribution of predicted probabilities — the primary drift/quality signal
# available in real time (label arrives days later, this arrives instantly).
PREDICTION_DISTRIBUTION = Histogram(
    "prediction_delay_probability", "Predicted delay probability distribution",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
)
MODEL_VERSION_TRAFFIC = Counter(
    "model_version_requests_total", "Requests served per model version", ["model_version"]
)
RISK_LEVEL_COUNT = Counter(
    "prediction_risk_level_total", "Predictions per risk level", ["risk_level"]
)

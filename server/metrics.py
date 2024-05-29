from fastapi import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    multiprocess,
)

LATENCY = Histogram(
    "http_server_requests_latency_ms",
    documentation="Time spent getting recommendations",
    buckets=(
        0.0001,
        0.0002,
        0.0005,
        0.001,
        0.01,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
        float("inf"),
    ),
)
REDIS_HITS = Counter(
    "redis_cache_hits_total",
    documentation="Hits in Redis Cache",
)
REDIS_MISS = Counter(
    "redis_cache_miss_total",
    documentation="Misses in Redis Cache",
)
REQUESTS = Counter(
    "http_server_requests_total",
    documentation="Number of requests",
)


def get_metrics() -> Response:
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    response = Response(generate_latest(registry=registry))
    response.headers["Content-Type"] = CONTENT_TYPE_LATEST
    return response

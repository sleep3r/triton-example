from typing import Callable

import numpy as np
from fastapi import Depends, FastAPI, Response, status
from pydantic import BaseModel, Field
from redis import Redis

from server.deps import get_redis, get_triton
from server.metrics import LATENCY, REDIS_HITS, REDIS_MISS, REQUESTS, get_metrics
from server.triton import TritonClient


class RecommendationsResponse(BaseModel):
    status: str = Field(
        default="success", description="Request status: success, error."
    )
    message: str | None = Field(
        default=None,
        description="Response message. It is present if an error occurs.",
    )
    payload: list[int] = Field(
        default=[], description="List of recommendations for user."
    )


def health() -> Response:
    return Response(content="ok", status_code=status.HTTP_200_OK)


@LATENCY.time()
def recommendations(
    model: str,
    user_id: int,
    response: Response,
    redis: Redis | None = Depends(get_redis),
    get_triton_model: Callable[[str], TritonClient | None] = Depends(get_triton),
) -> RecommendationsResponse:
    REQUESTS.inc()
    # 0. Check Redis and Triton
    if redis is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return RecommendationsResponse(status="error", message="redis unavailable")
    triton_model = get_triton_model(model)
    if triton_model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return RecommendationsResponse(status="error", message="triton unavailable")
    # 1. Check saved predictions in Redis
    redis_key = f"recs:{model}:{user_id}"
    recs = redis.lrange(redis_key, 0, -1)
    if len(recs) > 0:
        REDIS_HITS.inc()
        return RecommendationsResponse(payload=recs)
    # 2. Get recommendations from Triton model and save them
    REDIS_MISS.inc()
    output = triton_model.predict({"users": np.array([[user_id]])})
    recs = output.get("recommendations")
    if recs is None:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return RecommendationsResponse(
            status="error", message="triton failed to process request"
        )
    recs = recs[0, :10].tolist()
    redis.rpush(redis_key, *recs)
    return RecommendationsResponse(payload=recs)


def register_routes(app: FastAPI) -> None:
    # System information.
    app.add_api_route("/health", health, methods=["GET"], description="I'm alive!!!!")
    app.add_api_route(
        "/metrics", get_metrics, methods=["GET"], description="Get server metrics."
    )
    # Recommendation calls
    app.add_api_route(
        "/recommendations/{model}/{user_id}",
        recommendations,
        methods=["GET"],
        description="Get recommendations for user by user_id with model.",
        response_model=RecommendationsResponse,
    )

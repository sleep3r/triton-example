from typing import Callable

import numpy as np
from fastapi import Depends, FastAPI, Response, status
from pydantic import BaseModel, Field
from redis import Redis

from server.deps import get_redis, get_triton
from server.metrics import LATENCY, REDIS_HITS, REDIS_MISS, REQUESTS, get_metrics
from server.triton import TritonClient


class ModelResponse(BaseModel):
    status: str = Field(
        default="success", description="Request status: success, error."
    )
    message: str | None = Field(
        default=None,
        description="Response message. It is present if an error occurs.",
    )
    payload: list[int] = Field(
        default=[], description="Predicted class."
    )


def health() -> Response:
    return Response(content="ok", status_code=status.HTTP_200_OK)


@LATENCY.time()
def predict(
    response: Response,
    redis: Redis | None = Depends(get_redis),
    get_triton_model: Callable[[str], TritonClient | None] = Depends(get_triton),
) -> ModelResponse:
    REQUESTS.inc()
    # 0. Check Redis and Triton
    if redis is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ModelResponse(status="error", message="redis unavailable")

    triton_model = get_triton_model("clf")
    if triton_model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ModelResponse(status="error", message="triton unavailable")

    # 1. Check saved predictions in Redis
    redis_key = "123"
    recs = redis.lrange(redis_key, 0, -1)
    if len(recs) > 0:
        REDIS_HITS.inc()
        return ModelResponse(payload=recs)

    # 2. Get recommendations from Triton model and save them
    REDIS_MISS.inc()
    output = triton_model.predict({"users": np.array([[]])})
    recs = output.get("output_class")
    if recs is None:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ModelResponse(
            status="error", message="triton failed to process request"
        )
    recs = recs[0, :10].tolist()
    redis.rpush(redis_key, *recs)
    return ModelResponse(payload=recs)


def register_routes(app: FastAPI) -> None:
    # System information.
    app.add_api_route("/health", health, methods=["GET"], description="I'm alive!!!!")
    app.add_api_route(
        "/metrics", get_metrics, methods=["GET"], description="Get server metrics."
    )
    # Recommendation calls
    app.add_api_route(
        "/predict",
        predict,
        methods=["GET"],
        description="Get prediction from the model.",
        response_model=ModelResponse,
    )

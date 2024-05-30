import hashlib
from typing import Callable

from PIL import Image
from fastapi import Depends, FastAPI, File, Response, UploadFile, status
import numpy as np
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
        default=None, description="Response message. It is present if an error occurs."
    )
    payload: list[int] = Field(default=[], description="Predicted class.")


@LATENCY.time()
def predict(
    response: Response,
    image: UploadFile = File(...),
    redis: Redis | None = Depends(get_redis),
    get_triton_model: Callable[[str], TritonClient | None] = Depends(get_triton),
) -> ModelResponse:
    REQUESTS.inc()

    if redis is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ModelResponse(status="error", message="redis unavailable")

    triton_model = get_triton_model("clf")
    if triton_model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ModelResponse(status="error", message="triton unavailable")

    # Read image and generate hash
    img = Image.open(image.file)
    img_hash = hashlib.md5(img.tobytes()).hexdigest()
    redis_key = f"image:{img_hash}"

    # Check saved predictions in Redis
    recs = redis.lrange(redis_key, 0, -1)
    if recs:
        REDIS_HITS.inc()
        recs = [int(x) for x in recs]
        return ModelResponse(payload=recs)

    # Prepare image for Triton model
    img_np = np.array(img).astype(np.float32) / 255.0  # Normalize if needed
    img_np = np.expand_dims(img_np, axis=0)  # Add batch dimension if required

    # Get predictions from Triton model and save them
    REDIS_MISS.inc()
    output = triton_model.predict({"image": img_np})
    recs = output.get("output_class")
    if recs is None:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ModelResponse(status="error", message="triton failed to process request")
    recs = recs[0, :10].tolist()
    redis.rpush(redis_key, *recs)
    return ModelResponse(payload=recs)


def register_routes(app: FastAPI) -> None:
    app.add_api_route(
        "/health",
        Response(content="ok", status_code=status.HTTP_200_OK),
        methods=["GET"],
        description="I'm alive!!!!",
    )
    app.add_api_route(
        "/metrics", get_metrics, methods=["GET"], description="Get server metrics."
    )
    app.add_api_route(
        "/predict",
        predict,
        methods=["POST"],
        description="Get prediction from the model.",
        response_model=ModelResponse,
    )

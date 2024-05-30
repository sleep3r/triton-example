import hashlib
from typing import Callable, Union

from PIL import Image
from fastapi import Depends, FastAPI, File, Response, UploadFile, status
import numpy as np
from pydantic import BaseModel, Field
from redis import Redis
from loguru import logger

from server.deps import get_redis, get_triton
from server.triton import TritonClient


class CVResponse(BaseModel):
    status: str = Field(
        default="success", description="Request status: success, error."
    )
    message: str | None = Field(
        default=None, description="Response message. It is present if an error occurs."
    )
    payload: list[float] = Field(default=[], description="Predicted probas.")


async def predict(
        response: Response,
        model: str = "mnist",
        image: UploadFile = File(...),
        redis: Redis = Depends(get_redis),
        get_triton_model: Callable[[str], Union[TritonClient, None]] = Depends(get_triton),
) -> CVResponse:
    if redis is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return CVResponse(status="error", message="Redis unavailable")

    triton_model = get_triton_model(model)
    if triton_model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return CVResponse(status="error", message="Triton unavailable")

    # Read image and generate hash
    img = Image.open(image.file).convert("RGB")
    img_hash = hashlib.md5(img.tobytes()).hexdigest()
    redis_key = f"image:{img_hash}"

    # Check saved predictions in Redis
    probas = redis.lrange(redis_key, 0, -1)
    if probas:
        probas = [float(x) for x in probas]  # Convert Redis records to float
        return CVResponse(payload=probas)

    # Prepare image for Triton model
    img_np = np.array(img).astype(np.float32)
    logger.error(f"Image shape: {img_np.shape}")
    img_np = np.transpose(img_np, (2, 0, 1)) # Change image layout to CHW
    img_np = np.expand_dims(img_np, axis=0)  # Add batch dimension if required

    # Get predictions from Triton model and save them
    output = triton_model.predict({"input": img_np})
    logger.error(f"Output: {output}")

    if not output:  # Check if prediction is empty or None
        response.status_code = status.HTTP_400_BAD_REQUEST
        return CVResponse(status="error", message="Triton failed to process request")

    predict = output.get("output", np.array([[]])).astype(float).tolist()[0]
    logger.error(f"Predictions: {predict}")
    redis.rpush(redis_key, *map(str, predict))
    return CVResponse(payload=predict)


def health_check():
    logger.error("Health check")
    return Response(content="ok", status_code=status.HTTP_200_OK)


def register_routes(app: FastAPI) -> None:
    app.add_api_route(
        "/health",
        health_check,
        methods=["GET"],
        description="I'm alive!!!!",
    )
    app.add_api_route(
        "/predict",
        predict,
        methods=["POST"],
        description="Get prediction from the model.",
        response_model=CVResponse,
    )

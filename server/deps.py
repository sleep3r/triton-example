import os
from typing import Callable

from loguru import logger
from redis import Redis

from server.triton import TritonClient, TritonClientSettings


def get_redis() -> Redis | None:
    url = os.getenv("REDIS_URL")
    if url is None:
        return None
    redis = Redis.from_url(url)
    if not redis.ping():
        logger.error("redis unreachable")
        return None
    return redis


def get_triton() -> Callable[[str], TritonClient | None]:
    def wrapper(model: str) -> TritonClient | None:
        url = os.getenv("TRITON_URL")
        if url is None:
            return None
        settings = TritonClientSettings(url=url, model=model, version="1")
        client = TritonClient(settings, output_keys=["recommendations"])
        error, ok = client.ping()
        if not ok:
            logger.error(error)
            return None
        return client

    return wrapper

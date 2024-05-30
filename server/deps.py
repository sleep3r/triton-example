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


def get_triton() -> Callable[[str, str], TritonClient | None]:
    def wrapper(model: str = "clf", output: str = "output") -> TritonClient | None:
        url = os.getenv("TRITON_URL")
        if url is None:
            return None
        settings = TritonClientSettings(url=url, model=model, version="1")
        client = TritonClient(settings, output_keys=[output])
        error, ok = client.ping()
        if not ok:
            logger.error(error)
            return None
        return client

    return wrapper

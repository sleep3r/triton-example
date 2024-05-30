from fastapi import FastAPI

from server.api import register_routes


def create_app() -> FastAPI:
    app = FastAPI(title="Inference API")
    register_routes(app)
    return app

import os

from fastapi import FastAPI

from server.api import register_routes


def create_app() -> FastAPI:
    app = FastAPI(title="HSE-RecSys Server")
    register_routes(app)
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        create_app(), host="0.0.0.0", port=int(os.getenv("PORT", "1234")), debug=True
    )

import os

import uvicorn

from main import app as fastapi_app

app = fastapi_app


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        log_level="info",
    )

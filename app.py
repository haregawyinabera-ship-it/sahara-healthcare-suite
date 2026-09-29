import os

import gradio as gr
import uvicorn

from main import app as fastapi_app


with gr.Blocks(title="Sahara Healthcare Suite API") as space_ui:
    gr.Markdown("# Sahara Healthcare Suite")
    gr.Markdown(
        "FastAPI clinical services are available at the API routes. "
        "Open the [API documentation](/docs) or [health check](/health)."
    )

app = gr.mount_gradio_app(fastapi_app, space_ui, path="/")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860"))),
    )

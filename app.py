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

# Expose a Gradio app object for the Spaces runtime while keeping the mounted
# FastAPI backend available at the API routes.
demo = space_ui
app = gr.mount_gradio_app(fastapi_app, demo, path="/")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        log_level="info",
    )

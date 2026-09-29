import os

import gradio as gr
import spaces
import uvicorn

from main import app as fastapi_app


@spaces.GPU(duration=1)
def _gpu_probe():
    return "GPU attached"


with gr.Blocks(title="Sahara Healthcare Suite API") as space_ui:
    gr.Markdown("# Sahara Healthcare Suite")
    gr.Markdown(
        "FastAPI clinical services are available at the API routes. "
        "Open the [API documentation](/docs) or [health check](/health)."
    )

# Keep the FastAPI backend mounted under the Gradio Space entrypoint, while
# explicitly starting the web server so the process stays alive.
app = gr.mount_gradio_app(fastapi_app, space_ui, path="/")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        log_level="info",
    )

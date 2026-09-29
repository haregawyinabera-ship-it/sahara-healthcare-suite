import gradio as gr

from main import app as fastapi_app


with gr.Blocks(title="Sahara Healthcare Suite API") as space_ui:
    gr.Markdown("# Sahara Healthcare Suite")
    gr.Markdown(
        "FastAPI clinical services are available at the API routes. "
        "Open the [API documentation](/docs) or [health check](/health)."
    )

app = gr.mount_gradio_app(fastapi_app, space_ui, path="/")

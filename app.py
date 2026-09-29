import gradio as gr

from main import app as fastapi_app

with gr.Blocks(title="Sahara Healthcare Suite API") as space_ui:
    gr.Markdown("# Sahara Healthcare Suite")
    gr.Markdown(
        "FastAPI clinical services are available at the API routes. "
        "Open the [API documentation](/docs) or [health check](/health)."
    )

# Keep the FastAPI backend mounted under the Gradio Space entrypoint, but do not
# start a second Uvicorn server in the same process; the Spaces runtime already
# owns the listener lifecycle.
app = gr.mount_gradio_app(fastapi_app, space_ui, path="/")

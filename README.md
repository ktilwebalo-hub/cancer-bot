---
title: Cancer Awareness RAG
emoji: 🏥
colorFrom: pink
colorTo: red
sdk: gradio
sdk_version: 6.18.0
python_version: '3.11'
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Hugging Face deployment

This app uses a local knowledge-base retriever and OpenAI for generation.

Before launching the Space, add these secrets in Hugging Face:
- `OPENAI_API_KEY`

If you rebuild the knowledge base, run `uv run ingest.py` locally first so the cached chunks stay in sync with the markdown files.
# required
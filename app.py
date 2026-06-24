import gradio as gr
from dotenv import load_dotenv
from Implementation.answer import answer_question

load_dotenv(override=True)


def format_context(docs):
    """Format retrieved context for display."""
    if not docs:
        return "No retrieved context available. Please ensure the knowledge base is properly loaded."
    
    result = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("filename", "Unknown")
        doc_type = doc.metadata.get("doc_type", "General")
        # Truncate content for display if too long
        content = doc.page_content
        if len(content) > 500:
            content = content[:500] + "..."
        result += f"📄 **Document {i+1}:** {source} ({doc_type})\n\n{content}\n\n{'-'*50}\n\n"
    return result


def chat(message, history):
    """Handle user input and return assistant response."""
    history = history or []

    answer, docs = answer_question(message, history)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return "", history, format_context(docs)


def main():
    with gr.Blocks(title="CancerAware Expert Assistant") as ui:
        gr.Markdown("""
        # 🏥 CancerAware Expert Assistant

        Your compassionate guide to cancer awareness, prevention, and support.
        Ask me anything about:
        - 🩺 Cancer prevention and risk factors
        - 🔬 Screening and early detection
        - 💊 Treatment options
        - 🤝 Support resources
        - 💡 General cancer awareness

        *Disclaimer: I provide general information and guidance. Always consult healthcare professionals for medical advice.*
        """)

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=500,
                    avatar_images=("🧑", "🏥")
                )
                message = gr.Textbox(
                    placeholder="Ask about cancer prevention, screening, treatment, or support...",
                    show_label=False,
                    lines=2
                )
            with gr.Column(scale=1):
                context_box = gr.Textbox(
                    label="📚 Retrieved Knowledge Base",
                    lines=25,
                    interactive=False
                )

        # Fixed: Only use message.submit, not chatbot.submit
        message.submit(chat, inputs=[message, chatbot], outputs=[message, chatbot, context_box])

        gr.Markdown("""
        ---
        **How this works:** I search through a comprehensive cancer awareness knowledge base
        to provide you with accurate, evidence-based information. My responses are designed to be
        compassionate, practical, and supportive.
        """)

    ui.launch(theme=gr.themes.Soft(), css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .chatbot-container {
            border-radius: 12px;
            border: 1px solid #e0e0e0;
        }
    """)


if __name__ == "__main__":
    main()

#required
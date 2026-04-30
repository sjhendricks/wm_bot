import gradio as gr
from wm_bot_rag import WMChatbot

# Load once at startup (heavy — takes ~30s on GPU)
print("Loading W&M Chatbot...", flush=True)
bot = WMChatbot()
print("Ready!", flush=True)

def respond(message, history):
    reply = bot.chat(message, history)
    return reply

with gr.Blocks(theme=gr.themes.Soft(), title="W&M Academic Advisor") as demo:
    gr.Markdown(
        """
        # 🟢 William & Mary Academic Advisor
        Ask questions about courses, requirements, policies, and more.
        *Responses are based on official W&M catalog content.*
        """
    )
    chatbot = gr.Chatbot(height=450)
    msg = gr.Textbox(placeholder="e.g. What are the gen-ed requirements for COLL 200?", label="Your Question")
    clear = gr.Button("Clear")

    msg.submit(respond, [msg, chatbot], [chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Required for HPC port forwarding
        server_port=7860,
        share=False             # Set True for a public Gradio link (easier for testing)
    )
import os
from huggingface_hub import InferenceClient
import gradio as gr

# Initialize the client with the correct provider
HF_TOKEN = os.environ.get("HF_TOKEN", None)
client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)

def translate_text(text, src_lang, tgt_lang):
    """
    Function to handle translation using the MBART model
    """
    if not text.strip():
        return "Please enter text to translate"
    
    try:
        # Use the translation method with model-specific parameters
        result = client.translation(
            text,
            model="facebook/mbart-large-50-many-to-many-mmt",
            src_lang=src_lang,  # Source language code
            tgt_lang=tgt_lang   # Target language code
        )
        return result
        
    except Exception as e:
        return f"Error in translation: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Multilingual Translation Chatbot", theme="soft") as demo:
    gr.Markdown("# 🌍 Multilingual Translation Chatbot")
    gr.Markdown("Translate between 50 languages using facebook/mbart-large-50-many-to-many-mmt")
    
    with gr.Row():
        with gr.Column():
            src_lang = gr.Dropdown(
                choices=[
                    "ar_AR", "en_XX", "fr_XX", "es_XX", "de_DE", 
                    "zh_CN", "hi_IN", "ru_RU", "ja_XX", "pt_XX", "it_IT"
                ],
                value="ru_RU",
                label="Source Language Code"
            )
            input_text = gr.Textbox(
                lines=4,
                placeholder="Enter text to translate...",
                label="Input Text",
                value="Меня зовут Вольфганг и я живу в Берлине"
            )
        
        with gr.Column():
            tgt_lang = gr.Dropdown(
                choices=[
                    "ar_AR", "en_XX", "fr_XX", "es_XX", "de_DE", 
                    "zh_CN", "hi_IN", "ru_RU", "ja_XX", "pt_XX", "it_IT"
                ],
                value="en_XX",
                label="Target Language Code"
            )
            output_text = gr.Textbox(
                lines=4,
                label="Translation",
                interactive=False
            )
    
    # Translate button
    translate_btn = gr.Button("Translate 🌐", variant="primary")
    translate_btn.click(
        fn=translate_text,
        inputs=[input_text, src_lang, tgt_lang],
        outputs=output_text
    )
    
    # Examples for quick testing
    gr.Examples(
        examples=[
            ["Меня зовут Вольфганг и я живу в Берлине", "ru_RU", "en_XX"],
            ["Hello, how are you today?", "en_XX", "es_XX"],
            ["Bonjour, comment ça va?", "fr_XX", "en_XX"],
            ["今天天气很好", "zh_CN", "en_XX"]
        ],
        inputs=[input_text, src_lang, tgt_lang]
    )
    
    # Clear button
    clear_btn = gr.Button("Clear")
    clear_btn.click(
        fn=lambda: ["", "ru_RU", "en_XX", ""],
        outputs=[input_text, src_lang, tgt_lang, output_text]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)

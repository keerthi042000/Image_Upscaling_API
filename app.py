# import os
# from huggingface_hub import InferenceClient
# import gradio as gr

# # Initialize the client with the correct provider
# HF_TOKEN = os.environ.get("HF_TOKEN", None)
# client = InferenceClient(
#     provider="hf-inference",
#     api_key=HF_TOKEN
# )

# def translate_text(text, src_lang, tgt_lang):
#     """
#     Function to handle translation using the MBART model
#     """
#     if not text.strip():
#         return "Please enter text to translate"
    
#     try:
#         # Use the translation method with model-specific parameters
#         result = client.translation(
#             text,
#             model="facebook/mbart-large-50-many-to-many-mmt",
#             src_lang=src_lang,  # Source language code
#             tgt_lang=tgt_lang   # Target language code
#         )
#         return result
        
#     except Exception as e:
#         return f"Error in translation: {str(e)}"

# # Create the Gradio interface
# with gr.Blocks(title="Multilingual Translation Chatbot", theme="soft") as demo:
#     gr.Markdown("# 🌍 Multilingual Translation Chatbot")
#     gr.Markdown("Translate between 50 languages using facebook/mbart-large-50-many-to-many-mmt")
    
#     with gr.Row():
#         with gr.Column():
#             src_lang = gr.Dropdown(
#                 choices=[
#                     "ar_AR", "en_XX", "fr_XX", "es_XX", "de_DE", 
#                     "zh_CN", "hi_IN", "ru_RU", "ja_XX", "pt_XX", "it_IT"
#                 ],
#                 value="ru_RU",
#                 label="Source Language Code"
#             )
#             input_text = gr.Textbox(
#                 lines=4,
#                 placeholder="Enter text to translate...",
#                 label="Input Text",
#                 value="Меня зовут Вольфганг и я живу в Берлине"
#             )
        
#         with gr.Column():
#             tgt_lang = gr.Dropdown(
#                 choices=[
#                     "ar_AR", "en_XX", "fr_XX", "es_XX", "de_DE", 
#                     "zh_CN", "hi_IN", "ru_RU", "ja_XX", "pt_XX", "it_IT"
#                 ],
#                 value="en_XX",
#                 label="Target Language Code"
#             )
#             output_text = gr.Textbox(
#                 lines=4,
#                 label="Translation",
#                 interactive=False
#             )
    
#     # Translate button
#     translate_btn = gr.Button("Translate 🌐", variant="primary")
#     translate_btn.click(
#         fn=translate_text,
#         inputs=[input_text, src_lang, tgt_lang],
#         outputs=output_text
#     )
    
#     # Examples for quick testing
#     gr.Examples(
#         examples=[
#             ["Меня зовут Вольфганг и я живу в Берлине", "ru_RU", "en_XX"],
#             ["Hello, how are you today?", "en_XX", "es_XX"],
#             ["Bonjour, comment ça va?", "fr_XX", "en_XX"],
#             ["今天天气很好", "zh_CN", "en_XX"]
#         ],
#         inputs=[input_text, src_lang, tgt_lang]
#     )
    
#     # Clear button
#     clear_btn = gr.Button("Clear")
#     clear_btn.click(
#         fn=lambda: ["", "ru_RU", "en_XX", ""],
#         outputs=[input_text, src_lang, tgt_lang, output_text]
#     )

#     print("hellow")

# # Launch the interface
# if __name__ == "__main__":
#     demo.launch(share=True)




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
        return "Please enter any text to translate 😃"

    try:
        # Use the translation method with model-specific parameters
        src_code = lang_map[src_lang]
        tgt_code = lang_map[tgt_lang]

        result = client.translation(
            text,
            model="facebook/mbart-large-50-many-to-many-mmt",
            src_lang=src_code,
            tgt_lang=tgt_code
        )
        return result.translation_text

    except Exception as e:
        return f"Error in translation: {str(e)}"

custom_theme = gr.themes.Default().set(
    body_background_fill="#D3D3D3",   # light grey button 
    button_primary_background_fill="#000000",  # black background
    button_primary_text_color="#FFFFFF"        # white text
)


# Create the Gradio interface
with gr.Blocks(theme=custom_theme) as demo:
    gr.Markdown("# Multilingual Text Translator")


    lang_map = {
    "Arabic": "ar_AR",
    "Czech": "cs_CZ",
    "German": "de_DE",
    "English": "en_XX",
    "Spanish": "es_XX",
    "Estonian": "et_EE",
    "Finnish": "fi_FI",
    "French": "fr_XX",
    "Gujarati": "gu_IN",
    "Hindi": "hi_IN",
    "Italian": "it_IT",
    "Japanese": "ja_XX",
    "Kazakh": "kk_KZ",
    "Korean": "ko_KR",
    "Lithuanian": "lt_LT",
    "Latvian": "lv_LV",
    "Burmese": "my_MM",
    "Nepali": "ne_NP",
    "Dutch": "nl_XX",
    "Romanian": "ro_RO",
    "Russian": "ru_RU",
    "Sinhala": "si_LK",
    "Turkish": "tr_TR",
    "Vietnamese": "vi_VN",
    "Chinese": "zh_CN",
    "Afrikaans": "af_ZA",
    "Azerbaijani": "az_AZ",
    "Bengali": "bn_IN",
    "Persian": "fa_IR",
    "Hebrew": "he_IL",
    "Croatian": "hr_HR",
    "Indonesian": "id_ID",
    "Georgian": "ka_GE",
    "Khmer": "km_KH",
    "Macedonian": "mk_MK",
    "Malayalam": "ml_IN",
    "Mongolian": "mn_MN",
    "Marathi": "mr_IN",
    "Polish": "pl_PL",
    "Pashto": "ps_AF",
    "Portuguese": "pt_XX",
    "Swedish": "sv_SE",
    "Swahili": "sw_KE",
    "Tamil": "ta_IN",
    "Telugu": "te_IN",
    "Thai": "th_TH",
    "Tagalog": "tl_XX",
    "Ukrainian": "uk_UA",
    "Urdu": "ur_PK",
    "Xhosa": "xh_ZA",
    "Galician": "gl_ES",
    "Slovene": "sl_SI"
}


    with gr.Row():
        with gr.Column():
            src_lang = gr.Dropdown(
    choices=list(lang_map.keys()),
    value="English",     # now matches the choices
    label="Source Language"
)
            input_text = gr.Textbox(
                lines=4,
                placeholder="Enter text to translate...",
                label="Enter Text",
            )

        with gr.Column():
            tgt_lang = gr.Dropdown(
    choices=list(lang_map.keys()),
    value="English",
    label="Target Language"
)
            output_text = gr.Textbox(
                lines=4,
                label="Translation",
                interactive=False
            )

    # Translate button
    translate_btn = gr.Button("Translate ✨", elem_id="tr-btn", variant="primary")
    translate_btn.click(
        fn=translate_text,
        inputs=[input_text, src_lang, tgt_lang],
        outputs=output_text
    )
    
    # Clear button
    clear_btn = gr.Button("Clear", elem_id="clr-btn", variant="secondary")
    clear_btn.click(
      fn=lambda: ["", "English", "Russian", ""],
      outputs=[input_text, src_lang, tgt_lang, output_text]
    )

    print("hellow")

# Launch the interface
if _name_ == "_main_":
    demo.launch(share=True)

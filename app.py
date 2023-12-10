import gradio as gr
from inferenceTransformers import code_sequence as transformer
from inferenceBiLSTM import code_sequence as bilstm

LINES = 4

def translate_BiLSTM(text):
    # Your translation logic here
    translated_text = bilstm(text)
    # translated_text = f"Mô hình BiLSTM"
    return translated_text


def translate_transformer(text):
    # Your translation logic here
    translated_text = transformer(text)
    # translated_text = f"Mô hình Transformer"
    return translated_text

with gr.Blocks() as demo:
    title="Translation: English to Vietnamese"

    with gr.Tab("Seq2seq"):
        inp = gr.Textbox(placeholder="English", label="English", lines=LINES)
        out = gr.Textbox(label="Vietnamese", lines=LINES)
        btn = gr.Button("Translate")
        btn.click(fn=translate_BiLSTM, inputs=inp, outputs=out)

    with gr.Tab("Transformer"):
        inp = gr.Textbox(placeholder="English", label="English", lines=LINES)
        out = gr.Textbox(label="Vietnamese", lines=LINES)
        btn = gr.Button("Translate")
        btn.click(fn=translate_transformer, inputs=inp, outputs=out)

if __name__=="__main__":
    demo.launch()




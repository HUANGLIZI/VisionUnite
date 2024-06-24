import gradio as gr
import argparse

import ImageBind.data as data
import llama
import torch
import torchvision.transforms as transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="/output_dir/checkpoint-10-8-29.pth", type=str,
    help="Name of or path to fine-tune checkpoint",
)
args = parser.parse_args()

llama_dir = "/cpfs01/user/lizihan/llama-adapter/llama_model_weights"
model = llama.load(args.model, llama_dir)
print(model)
# model.half()
model.eval()

def load_and_transform_vision_data(image_paths, device):
    if image_paths is None:
        return None

    image_ouputs = []
    for image_path in image_paths:
        data_transform = transforms.Compose(
            [
                transforms.Resize(
                    448, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_ouputs.append(image)
    return torch.stack(image_ouputs, dim=0)


def caption_generate(
    img: str,
    prompt: str,
    # model,
    max_gen_len=128,
    temperature: float = 0.1,
    top_p: float = 0.75,
):
    input = None
    input_type = None
    
    try:
        input = load_and_transform_vision_data([img], device='cuda')
        input_type = 'vision'
        print('image', input.shape)
    except:
        pass
    
    prompts = [llama.format_prompt(prompt)]

    results = model.generate(input, prompts, input_type, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
    result = results[0][0]
    print(result)
    return result

def create_caption_demo():
    with gr.Blocks() as instruct_demo:
        with gr.Row():
            with gr.Column():
                question = gr.Textbox(lines=2, label="Question")
                img = gr.Image(label='Image', type='filepath')
                max_len = gr.Slider(minimum=1, maximum=512,
                                    value=128, label="Max length")
                with gr.Accordion(label='Advanced options', open=False):
                    temp = gr.Slider(minimum=0, maximum=1,
                                     value=0.1, label="Temperature")
                    top_p = gr.Slider(minimum=0, maximum=1,
                                      value=0.75, label="Top p")

                run_botton = gr.Button("Run")

            with gr.Column():
                outputs = gr.Textbox(lines=10, label="Output")

        inputs = [img, question, max_len, temp, top_p]

        examples = [
            ["/cpfs01/user/lizihan/llama-adapter/imagebind-llm/example_imgs/funny-photo.jpg",  "Explain why this image is funny", 128, 0.1, 0.75],
        ]

        gr.Examples(
            examples=examples,
            inputs=inputs,
            outputs=outputs,
            fn=caption_generate,
            cache_examples=False
        )
        run_botton.click(fn=caption_generate, inputs=inputs, outputs=outputs)
    return instruct_demo


description = f"""
# VisionUnite
"""

with gr.Blocks(css="h1,p {text-align: center;}") as demo:
    gr.Markdown(description)
    with gr.TabItem("Multi-Modal Interaction"):
        create_caption_demo()

demo.queue(api_open=True, concurrency_count=1).launch(share=True)

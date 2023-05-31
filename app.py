from PIL import Image
from collections import OrderedDict

import torch
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import gradio as gr

from model import Glow

def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes

model = Glow(
    1, 4, 3, affine=False, conv_lu=not False
)
state_dict = torch.load('model_best.pt', map_location='cpu')

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # module字段在最前面，从第7个字符开始就可以去掉module
    new_state_dict[name] = v  # 新字典的key值对应的value一一对应

model.load_state_dict(new_state_dict)
model = model.eval()


def generate_handwrite(model_option):
    with torch.no_grad():
        if model_option == 'glow':
            z_sample = []
            z_shapes = calc_z_shapes(1, 56, 4, 3)
            for z in z_shapes:
                z_new = torch.randn(64, *z) * 0.5
                z_sample.append(z_new)
            sample = model.reverse(z_sample).cpu().data
            sample = sample.view(64, 1, 56, 56)
            img = make_grid(sample, nrow=8, padding=2)
            img = to_pil_image(img)
            return img
        else:
            return Image.open('./sample_gt.png')


# generate_handwrite()
MODEL_LIST = ['glow', 'groundtruth']
with gr.Blocks() as demo:
    output = gr.Image(label="Output")
    model_option = gr.Radio(choices=MODEL_LIST, label="选择模型", value="groundtruth")
    greet_btn = gr.Button("generate handwrite number")
    greet_btn.click(fn=generate_handwrite, inputs=model_option, outputs=output, api_name="generate_handwrite")

demo.launch()

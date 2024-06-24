import torch
from collections import OrderedDict
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ori", required=True, type=str,
    help="Name of or path to pretrained checkpoint",
)
parser.add_argument(
    "--target", default=None,
    help="target position for the ckpt",
)
args = parser.parse_args()

ori_ckpt_path = Path(args.ori)
target_ckpt_path = ori_ckpt_path.with_stem("converted_" + ori_ckpt_path.stem)

ckpt = torch.load(ori_ckpt_path, map_location='cpu')

new_key_set = []
discarded  = []
for key in ckpt['model'].keys():
    if key.startswith("image_bind."):
        discarded.append(key)
    else:
        new_key_set.append(key)

discarded1 = []
new_key_set1 = []
for key in new_key_set:
    if key.startswith("llma.") and "bias" not in key and "gate" not in key and "lora" not in key and "norm" not in key:
        discarded1.append(key)
    else:
        new_key_set1.append(key)

new_key_set1.remove('prefix_projector_norm.weight')
new_key_set1.remove('prefix_projector_norm.bias')

new_ckpt = {'model': OrderedDict()}

for key in new_key_set1:
    new_ckpt['model'][key.replace("llma", "llama")] = ckpt['model'][key]

torch.save(new_ckpt, target_ckpt_path)
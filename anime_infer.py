# ================ 你原来的代码保持不变 ================
import os
import argparse
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from model import Generator

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def load_image(image_path, max_edge=1024, x32=True):
    img = Image.open(image_path).convert("RGB")
    ow, oh = img.size
    scale = min(max_edge / ow, max_edge / oh, 1.0)
    nw, nh = int(ow * scale), int(oh * scale)
    if x32:
        nw = (nw // 32) * 32
        nh = (nh // 32) * 32
    if (nw, nh) != (ow, oh):
        img = img.resize((nw, nh), Image.LANCZOS)
    return img, (ow, oh)

def test(args):
    device = args.device
    net = Generator()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net.to(device).eval()
    os.makedirs(args.output_dir, exist_ok=True)
    for image_name in sorted(os.listdir(args.input_dir)):
        if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            continue
        image_path = os.path.join(args.input_dir, image_name)
        image, (ow, oh) = load_image(image_path, max_edge=1024, x32=True)
        with torch.no_grad():
            x = to_tensor(image).unsqueeze(0) * 2 - 1
            out = net(x.to(device), args.upsample_align).cpu()
            out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
            out = torch.nn.functional.interpolate(
                out.unsqueeze(0), size=(oh, ow), mode="bilinear", align_corners=False
            ).squeeze(0)
            out_img = to_pil_image(out)
        out_img.save(os.path.join(args.output_dir, image_name))

# ================ 新增：供 GUI 直接调用 ================
def run_infer(checkpoint: str, input_dir: str, output_dir: str,
              device: str = 'cpu', upsample_align: bool = False):
    """GUI 专用入口，不再起子进程"""
    import argparse
    args = argparse.Namespace()
    args.checkpoint      = checkpoint
    args.input_dir       = input_dir
    args.output_dir      = output_dir
    args.device          = device
    args.upsample_align  = upsample_align
    test(args)

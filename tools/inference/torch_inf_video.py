"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

torch_inf_video.py: 動画入力 → BBox描画 → 動画出力
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

COLORS = plt.cm.tab20.colors


def load_label_map(label_file=None):
    if label_file and os.path.exists(label_file):
        with open(label_file) as f:
            names = [l.strip() for l in f if l.strip()]
        return {i: name for i, name in enumerate(names)}
    return {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorbike', 5: 'aeroplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'trafficlight',
        11: 'firehydrant', 12: 'streetsign', 13: 'stopsign', 14: 'parkingmeter',
        15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse',
        20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
        25: 'giraffe', 26: 'hat', 27: 'backpack', 28: 'umbrella', 29: 'shoe',
        30: 'eyeglasses', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
        35: 'skis', 36: 'snowboard', 37: 'sportsball', 38: 'kite', 39: 'baseballbat',
        40: 'baseballglove', 41: 'skateboard', 42: 'surfboard', 43: 'tennisracket',
        44: 'bottle', 45: 'plate', 46: 'wineglass', 47: 'cup', 48: 'fork',
        49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
        54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hotdog',
        59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'sofa',
        64: 'pottedplant', 65: 'bed', 66: 'mirror', 67: 'diningtable', 68: 'window',
        69: 'desk', 70: 'toilet', 71: 'door', 72: 'tv', 73: 'laptop',
        74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cellphone', 78: 'microwave',
        79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 83: 'blender',
        84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddybear',
        89: 'hairdrier', 90: 'toothbrush', 91: 'hairbrush'
    }


label_map = {}
COLOR_MAP = {}


def draw_on_image(im_pil, labels, boxes, scores, thrh=0.45):
    draw = ImageDraw.Draw(im_pil)
    font = ImageFont.load_default()
    mask = scores > thrh
    labels, boxes, scores = labels[mask], boxes[mask], scores[mask]

    for j, box in enumerate(boxes):
        category = labels[j].item()
        color = COLOR_MAP.get(category, (255, 255, 255))
        box = list(map(int, box))

        draw.rectangle(box, outline=color, width=3)

        text = f"{label_map[category]} {scores[j].item():.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        draw.rectangle([box[0], box[1] - th - 2, box[0] + tw + 4, box[1]], fill=color)
        draw.text((box[0] + 2, box[1] - th - 2), text, fill="black", font=font)

    return im_pil


def process_video(model, input_path, output_path, thrh=0.45, size=(640, 640), vit_backbone=False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {input_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if vit_backbone else T.Lambda(lambda x: x)
    ])

    print(f"入力: {input_path}")
    print(f"解像度: {orig_w}x{orig_h}  FPS: {fps:.1f}  総フレーム: {total}")
    print(f"出力: {output_path}")

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # BGR(cv2) → RGB(PIL)
        im_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        orig_size = torch.tensor([[orig_w, orig_h]]).cuda()
        im_data = transforms(im_pil).unsqueeze(0).cuda()

        with torch.no_grad():
            output = model(im_data, orig_size)

        labels = output[0]['labels']
        boxes  = output[0]['boxes']
        scores = output[0]['scores']

        vis_pil = draw_on_image(im_pil.copy(), labels, boxes, scores, thrh)

        # PIL(RGB) → BGR(cv2) → VideoWriter
        vis_bgr = cv2.cvtColor(np.array(vis_pil), cv2.COLOR_RGB2BGR)
        writer.write(vis_bgr)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total} フレーム処理済み...")

    cap.release()
    writer.release()
    print(f"\n完了: {frame_idx} フレームを処理しました。")
    print(f"保存先: {output_path}")


def main(args):
    global label_map, COLOR_MAP
    label_map = load_label_map(args.label_file)
    COLOR_MAP = {
        label: tuple([int(c * 255) for c in COLORS[i % len(COLORS)]])
        for i, label in enumerate(label_map)
    }

    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    checkpoint = torch.load(args.resume, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.eval().cuda()
            self.postprocessor = cfg.postprocessor.eval().cuda()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model()
    img_size = cfg.yaml_cfg["eval_spatial_size"]
    vit_backbone = cfg.yaml_cfg.get('DINOv3STAs', False)

    process_video(model, args.input, args.output, thrh=args.threshold,
                  size=img_size, vit_backbone=vit_backbone)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',     type=str, required=True,  help='設定ファイル (.yml)')
    parser.add_argument('-r', '--resume',     type=str, required=True,  help='チェックポイント (.pth)')
    parser.add_argument('-i', '--input',      type=str, required=True,  help='入力動画パス')
    parser.add_argument('-o', '--output',     type=str, required=True,  help='出力動画パス (.mp4)')
    parser.add_argument('-l', '--label_file', type=str, default=None,   help='クラス名ファイル (label.txt)')
    parser.add_argument('--threshold',        type=float, default=0.45, help='スコア閾値 (default: 0.45)')
    args = parser.parse_args()
    main(args)

"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

torch_inf_vis2.py: 推論速度・GPUメモリ計測機能追加版
"""

import os
import sys
import time

import cv2  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np  # noqa: F401
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

COLORS = plt.cm.tab20.colors


def load_label_map(label_file=None):
    """label_file が指定されればそこから読む (0-indexed)。なければ COCO デフォルト。"""
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


def draw(image, labels, boxes, scores, thrh=0.45):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    labels, boxes, scores = labels[scores > thrh], boxes[scores > thrh], scores[scores > thrh]

    for j, box in enumerate(boxes):
        category = labels[j].item()
        color = COLOR_MAP.get(category, (255, 255, 255))
        box = list(map(int, box))

        draw.rectangle(box, outline=color, width=3)

        text = f"{label_map[category]} {scores[j].item():.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        text_background = [box[0], box[1] - text_height - 2, box[0] + text_width + 4, box[1]]
        draw.rectangle(text_background, fill=color)
        draw.text((box[0] + 2, box[1] - text_height - 2), text, fill="black", font=font)

    return image


def process_dataset(model, dataset_path, output_path, thrh=0.5, size=(640, 640),
                    vit_backbone=False, ann_file=None, warmup=10, batch_size=1):
    """
    Parameters
    ----------
    warmup : int
        GPUウォームアップのため最初の N バッチは計測対象から除外する数。
    batch_size : int
        推論時のバッチサイズ。レイテンシは「バッチ処理時間 / バッチサイズ」で1枚あたりに換算。
    """
    import json
    os.makedirs(output_path, exist_ok=True)

    if ann_file and os.path.exists(ann_file):
        with open(ann_file) as f:
            coco = json.load(f)
        image_paths = []
        for img_info in coco['images']:
            fname = os.path.basename(img_info['file_name'])
            p = os.path.join(dataset_path, fname)
            if os.path.exists(p):
                image_paths.append(p)
    else:
        image_paths = [
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if f.endswith(('.jpg', '.png'))
        ]

    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if vit_backbone else T.Lambda(lambda x: x)
    ])

    print(f"Found {len(image_paths)} images...")
    print(f"Batch size: {batch_size}  |  Warmup: first {warmup} batches excluded from timing.")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # 1枚あたりのレイテンシ（ms）を蓄積
    per_image_latencies_ms = []
    processed = 0

    # バッチ単位で処理
    for batch_idx in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_idx: batch_idx + batch_size]
        pil_images = [Image.open(p).convert('RGB') for p in batch_paths]

        # orig_size: [B, 2]
        orig_sizes = torch.tensor([[im.width, im.height] for im in pil_images]).cuda()
        # im_data: [B, C, H, W]
        im_data = torch.stack([transforms(im) for im in pil_images]).cuda()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start_event.record()
            outputs = model(im_data, orig_sizes)
            end_event.record()

        torch.cuda.synchronize()
        batch_ms = start_event.elapsed_time(end_event)
        per_image_ms = batch_ms / len(batch_paths)

        current_batch_idx = batch_idx // batch_size
        if current_batch_idx >= warmup:
            per_image_latencies_ms.append(per_image_ms)

        # バッチ内の各画像を可視化保存
        for i, (im_pil, file_path) in enumerate(zip(pil_images, batch_paths)):
            labels = outputs[i]['labels']
            boxes  = outputs[i]['boxes']
            scores = outputs[i]['scores']
            vis_image = draw(im_pil.copy(), labels, boxes, scores, thrh)
            save_path = os.path.join(output_path, f"vis_{os.path.basename(file_path)}")
            vis_image.save(save_path)

        processed += len(batch_paths)
        if processed % 500 < batch_size:
            print(f"Processed {processed}/{len(image_paths)} images...")

    print("Visualization complete. Results saved in:", output_path)

    # ---- 計測結果サマリ ----
    peak_alloc_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024 ** 2

    print("\n" + "=" * 55)
    print("  Inference Performance Summary")
    print("=" * 55)
    print(f"  Batch size       : {batch_size}")
    if per_image_latencies_ms:
        measured_images = len(per_image_latencies_ms) * batch_size
        avg_ms = sum(per_image_latencies_ms) / len(per_image_latencies_ms)
        min_ms = min(per_image_latencies_ms)
        max_ms = max(per_image_latencies_ms)
        fps = 1000.0 / avg_ms
        print(f"  Measured images  : ~{measured_images}  (warmup={warmup} batches)")
        print(f"  Latency/img avg  : {avg_ms:.2f} ms")
        print(f"  Latency/img min  : {min_ms:.2f} ms")
        print(f"  Latency/img max  : {max_ms:.2f} ms")
        print(f"  Throughput (FPS) : {fps:.1f}")
    else:
        print("  (計測対象バッチが warmup 数以下のため、速度計測なし)")
    print(f"  GPU Memory allocated (peak) : {peak_alloc_mb:.1f} MB  ({peak_alloc_mb/1024:.2f} GB)")
    print(f"  GPU Memory reserved  (peak) : {peak_reserved_mb:.1f} MB  ({peak_reserved_mb/1024:.2f} GB)")
    print("=" * 55)


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

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.eval().cuda()
            self.postprocessor = cfg.postprocessor.eval().cuda()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()
    img_size = cfg.yaml_cfg["eval_spatial_size"]
    vit_backbone = cfg.yaml_cfg.get('DINOv3STAs', False)

    process_dataset(
        model, args.dataset, args.output,
        thrh=args.threshold, size=img_size,
        vit_backbone=vit_backbone, ann_file=args.ann_file,
        warmup=args.warmup, batch_size=args.batch_size,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',     type=str, required=True)
    parser.add_argument('-r', '--resume',     type=str, required=True)
    parser.add_argument('-d', '--dataset',    type=str, default='./data/fiftyone/validation/data')
    parser.add_argument('-o', '--output',     type=str, required=True, help="可視化結果の保存先")
    parser.add_argument('-l', '--label_file', type=str, default=None,  help="クラス名ファイル (label.txt, 0-indexed)")
    parser.add_argument('-a', '--ann_file',   type=str, default=None,  help="COCO アノテーション JSON（指定するとその画像のみ処理）")
    parser.add_argument('--threshold',        type=float, default=0.45, help="スコア閾値 (default: 0.45)")
    parser.add_argument('--warmup',           type=int,   default=10,   help="速度計測から除外するウォームアップバッチ数 (default: 10)")
    parser.add_argument('--batch_size',       type=int,   default=1,    help="推論バッチサイズ (default: 1)")
    args = parser.parse_args()
    main(args)

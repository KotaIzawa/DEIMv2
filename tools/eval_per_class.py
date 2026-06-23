"""
DEIMv2 定量評価スクリプト
per-class: TP / FP / FN / Precision / Recall / F1
overall  : mAP@50 / mAP@[50:95]
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from engine.core import YAMLConfig


# ------------------------------------------------------------------ #
#  モデルロード
# ------------------------------------------------------------------ #
def load_model(config_path, checkpoint_path, device):
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().to(device)
    model.eval()
    img_size = cfg.yaml_cfg.get('eval_spatial_size', [640, 640])
    return model, img_size


# ------------------------------------------------------------------ #
#  IoU 計算 (xyxy)
# ------------------------------------------------------------------ #
def box_iou(boxes1, boxes2):
    """boxes1: (N,4), boxes2: (M,4) → (N,M)"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


# ------------------------------------------------------------------ #
#  xywh (COCO) → xyxy
# ------------------------------------------------------------------ #
def xywh_to_xyxy(boxes):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([x, y, x + w, y + h], dim=1)


# ------------------------------------------------------------------ #
#  メイン
# ------------------------------------------------------------------ #
def main(args):
    device = torch.device(args.device)

    # ラベル読み込み
    with open(args.label_file) as f:
        class_names = [l.strip() for l in f if l.strip()]
    num_classes = len(class_names)
    print(f"クラス: {class_names}")

    # アノテーション読み込み
    with open(args.ann_file) as f:
        coco = json.load(f)

    # category_id → 0-indexed label のマッピング
    cat_ids = sorted([c['id'] for c in coco['categories']])
    cat_id_to_label = {cid: i for i, cid in enumerate(cat_ids)}

    # image_id → annotations
    gt_by_image = defaultdict(list)
    for ann in coco['annotations']:
        gt_by_image[ann['image_id']].append(ann)

    # image_id → filename
    id_to_info = {img['id']: img for img in coco['images']}

    # モデルロード
    print("モデルをロード中...")
    model, img_size = load_model(args.config, args.resume, device)
    print(f"入力サイズ: {img_size}")

    transforms = T.Compose([T.Resize(img_size), T.ToTensor()])

    # per-class 集計
    tp_list = defaultdict(list)   # (score, matched)
    fp_list = defaultdict(list)   # (score, not matched)
    fn_count = defaultdict(int)   # FN per class

    # COCO eval 用
    coco_preds = []

    for img_id, img_info in tqdm(id_to_info.items(), desc="推論中"):
        fname = img_info['file_name']
        img_path = os.path.join(args.img_dir, fname)
        if not os.path.exists(img_path):
            # ファイル名だけで再試行
            img_path = os.path.join(args.img_dir, os.path.basename(fname))
        if not os.path.exists(img_path):
            continue

        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        im_data = transforms(im_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_labels, pred_boxes, pred_scores = model(im_data, orig_size)

        pred_labels = pred_labels[0].cpu()
        pred_boxes  = pred_boxes[0].cpu()
        pred_scores = pred_scores[0].cpu()

        # 閾値フィルタ
        keep = pred_scores >= args.threshold
        pred_labels = pred_labels[keep]
        pred_boxes  = pred_boxes[keep]
        pred_scores = pred_scores[keep]

        # COCO preds 蓄積 (xyxy → xywh)
        for l, b, s in zip(pred_labels.tolist(), pred_boxes.tolist(), pred_scores.tolist()):
            x1, y1, x2, y2 = b
            coco_preds.append({
                'image_id': img_id,
                'category_id': cat_ids[int(l)],
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': s,
            })

        # GT
        anns = gt_by_image[img_id]
        gt_boxes_by_cls = defaultdict(list)
        for ann in anns:
            cls = cat_id_to_label[ann['category_id']]
            box = ann['bbox']  # xywh
            gt_boxes_by_cls[cls].append(box)

        # TP/FP/FN 計算 (per class, IoU≥iou_thresh)
        iou_thresh = args.iou_thresh
        for cls in range(num_classes):
            pred_mask = (pred_labels == cls)
            p_boxes = pred_boxes[pred_mask]
            p_scores = pred_scores[pred_mask]

            g_boxes_raw = gt_boxes_by_cls[cls]
            if len(g_boxes_raw) > 0:
                g_boxes = xywh_to_xyxy(torch.tensor(g_boxes_raw, dtype=torch.float32))
            else:
                g_boxes = torch.zeros((0, 4))

            matched_gt = set()

            # スコア降順でマッチング
            order = p_scores.argsort(descending=True)
            for idx in order:
                pb = p_boxes[idx].unsqueeze(0)
                sc = p_scores[idx].item()
                if len(g_boxes) > 0:
                    ious = box_iou(pb, g_boxes)[0]
                    best_iou, best_j = ious.max(0) if len(g_boxes) > 0 else (torch.tensor(0.), torch.tensor(0))
                    best_iou = best_iou.item()
                    best_j = best_j.item()
                    if best_iou >= iou_thresh and best_j not in matched_gt:
                        tp_list[cls].append(sc)
                        matched_gt.add(best_j)
                    else:
                        fp_list[cls].append(sc)
                else:
                    fp_list[cls].append(sc)

            fn_count[cls] += len(g_boxes) - len(matched_gt)

    # -------------------------------------------------------------- #
    #  Precision / Recall / F1 / TP / FP / FN の表示
    # -------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print(f"{'クラス':<12} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("=" * 70)

    total_tp = total_fp = total_fn = 0
    for cls in range(num_classes):
        tp = len(tp_list[cls])
        fp = len(fp_list[cls])
        fn = fn_count[cls]
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        total_tp += tp; total_fp += fp; total_fn += fn
        print(f"{class_names[cls]:<12} {tp:>6} {fp:>6} {fn:>6} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}")

    print("-" * 70)
    g_prec = total_tp / (total_tp + total_fp + 1e-9)
    g_rec  = total_tp / (total_tp + total_fn + 1e-9)
    g_f1   = 2 * g_prec * g_rec / (g_prec + g_rec + 1e-9)
    print(f"{'合計 (macro)':<12} {total_tp:>6} {total_fp:>6} {total_fn:>6} {g_prec:>8.4f} {g_rec:>8.4f} {g_f1:>8.4f}")
    print("=" * 70)

    # -------------------------------------------------------------- #
    #  mAP (COCO eval)
    # -------------------------------------------------------------- #
    try:
        from faster_coco_eval import COCO, COCOeval_faster
        coco_gt = COCO(args.ann_file)
        if coco_preds:
            coco_dt = coco_gt.loadRes(coco_preds)
            evaluator = COCOeval_faster(coco_gt, coco_dt, iouType='bbox', print_function=print, separate_eval=True)
            evaluator.evaluate()
            evaluator.accumulate()
            print("\n--- COCO mAP ---")
            evaluator.summarize()
    except Exception as e:
        print(f"\n[COCO eval スキップ] {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEIMv2 定量評価')
    parser.add_argument('-c', '--config',     type=str, required=True,  help='設定ファイル (.yml)')
    parser.add_argument('-r', '--resume',     type=str, required=True,  help='チェックポイント (.pth)')
    parser.add_argument('-a', '--ann_file',   type=str, required=True,  help='COCO アノテーション JSON')
    parser.add_argument('-i', '--img_dir',    type=str, required=True,  help='画像フォルダ')
    parser.add_argument('-l', '--label_file', type=str, required=True,  help='クラス名ファイル (label.txt)')
    parser.add_argument('--threshold', type=float, default=0.45,  help='スコア閾値 (default: 0.3)')
    parser.add_argument('--iou_thresh', type=float, default=0.5,  help='TP判定のIoU閾値 (default: 0.5)')
    parser.add_argument('-d', '--device',     type=str, default='cuda', help='デバイス')
    args = parser.parse_args()
    main(args)

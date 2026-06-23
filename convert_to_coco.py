import os
import json
from PIL import Image  # 実際の画像からサイズを測るために追加

# --- パス設定 ---
DATA_DIR = './my_dataset/data'
CFG_DIR = './my_dataset/cfg'
OUT_DIR = './my_dataset/annotations'

os.makedirs(OUT_DIR, exist_ok=True)

# クラスラベルの設定
categories = []
label_path = os.path.join(CFG_DIR, 'label.txt')
if os.path.exists(label_path):
    with open(label_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            class_name = line.strip()
            if class_name:
                categories.append({'id': idx + 1, 'name': class_name})
else:
    categories = [
        {'id': 1, 'name': 'person'},
        {'id': 2, 'name': 'stepladder'}
    ]

def create_coco_json(txt_filename, out_filename, min_confidence=0.0, exclude_pseudo=False):
    txt_path = os.path.join(CFG_DIR, txt_filename)
    if not os.path.exists(txt_path):
        return

    coco_format = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    ann_id = 1
    img_id = 1
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        base_name = line.strip()
        if not base_name: continue
        
        if base_name.endswith('.jpg'):
            base_name = base_name[:-4]
            
        img_filename = f"{base_name}.jpg"
        json_filename = f"{base_name}.json"
        
        img_path = os.path.join(DATA_DIR, img_filename)
        json_path = os.path.join(DATA_DIR, json_filename)
        
        if not os.path.exists(img_path) or not os.path.exists(json_path):
            continue
            
        # ★変更点: 実際の画像ファイルから直接サイズ（幅・高さ）を取得する
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        # 画像情報は、物体が写っていてもいなくても必ず登録する！
        coco_format["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "width": img_width,
            "height": img_height
        })
        
        # JSONを読み込み
        with open(json_path, 'r') as jf:
            single_ann_list = json.load(jf)
            
        # アノテーション（物体）が存在する場合のみ、ボックスの情報を追加する
        if isinstance(single_ann_list, list) and len(single_ann_list) > 0:
            for obj in single_ann_list:
                if exclude_pseudo and 'confidence_score' in obj:
                    continue
                conf = obj.get('confidence_score', 1.0)
                if conf < min_confidence:
                    continue
                bbox = obj['bbox']
                cat_id = obj['category_id'] + 1
                area = bbox[2] * bbox[3]
                
                coco_format["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id, 
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0
                })
                ann_id += 1
                
        img_id += 1
        
    with open(os.path.join(OUT_DIR, out_filename), 'w') as out_f:
        json.dump(coco_format, out_f, indent=4)
    mode = "疑似ラベル除外" if exclude_pseudo else f"confidence>={min_confidence}"
    print(f"{out_filename} 完了 (画像数: {img_id-1}, アノテーション数: {ann_id-1}, {mode})")

# 通常版（全アノテーション）
print("データ変換を開始します... (全アノテーション)")
create_coco_json('train.txt', 'train_coco.json')
create_coco_json('val.txt', 'val_coco.json')
create_coco_json('test.txt', 'test_coco.json')

# クリーン版（confidence_scoreフィールドを持つ疑似ラベルを除外）
print("\nクリーン版（疑似ラベル除外）を作成します...")
create_coco_json('train.txt', 'train_coco_clean.json', exclude_pseudo=True)
create_coco_json('val.txt', 'val_coco_clean.json', exclude_pseudo=True)
create_coco_json('test.txt', 'test_coco_clean.json', exclude_pseudo=True)
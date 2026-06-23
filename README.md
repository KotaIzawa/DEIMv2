# DEIMv2 - ACES カスタム学習・評価ガイド

DEIMv2をACESのデータセットで学習・評価するためのガイドです。

---

## S3リソース構成

```
s3://aces-project-denver-11/
├── dataset/
│   └── （画像データ .jpg, YOLO用 .json）
│
├── datasets/
│   └── new-domain-phase11/
│       └── annotations/
│           ├── train_coco_clean.json
│           ├── val_coco_clean.json
│           └── test_coco_clean.json
│
└── models/
    └── deimv2_m_phase11_20260611/
        ├── weights/
        │   └── best_stg2.pth
        └── configs/
            ├── deimv2_hgnetv2_m_phase11.yml
            ├── phase11_detection.yml
            ├── runtime.yml
            └── base/
                ├── dataloader.yml
                ├── optimizer.yml
                └── deimv2.yml
```

---

## 環境セットアップ

```bash
git clone https://github.com/KotaIzawa/DEIMv2.git
cd DEIMv2
pip install -r requirements.txt
```

---

## S3からファイルを取得・配置

git clone後、以下のコマンドでファイルをダウンロードして所定の場所に配置してください。

```bash
# ① モデルの重み → DEIMv2/outputs/ 以下に配置
mkdir -p outputs/deimv2_m_phase11_20260611
aws s3 cp s3://aces-project-denver-11/models/deimv2_m_phase11_20260611/weights/best_stg2.pth \
  outputs/deimv2_m_phase11_20260611/best_stg2.pth

# ② configファイル → DEIMv2/configs/ 以下に上書き配置
aws s3 sync s3://aces-project-denver-11/models/deimv2_m_phase11_20260611/configs/ \
  configs/

# ③ アノテーションJSON → 任意の場所に配置（例: /datadrive/annotations/）
mkdir -p /datadrive/annotations
aws s3 sync s3://aces-project-denver-11/datasets/new-domain-phase11/annotations/ \
  /datadrive/annotations/

# ④ 画像データ → 任意の場所に配置（例: /datadrive/data/）
mkdir -p /datadrive/data
aws s3 sync s3://aces-project-denver-11/dataset/ \
  /datadrive/data/
```

ダウンロード後のリポジトリ構成：

```
DEIMv2/
├── configs/
│   ├── deimv2/
│   │   └── deimv2_hgnetv2_m_phase11.yml   ← ② でダウンロード
│   ├── dataset/
│   │   └── phase11_detection.yml           ← ② でダウンロード
│   ├── runtime.yml                         ← ② でダウンロード
│   └── base/
│       ├── dataloader.yml                  ← ② でダウンロード
│       ├── optimizer.yml                   ← ② でダウンロード
│       └── deimv2.yml                      ← ② でダウンロード
└── outputs/
    └── deimv2_m_phase11_20260611/
        └── best_stg2.pth                   ← ① でダウンロード

/datadrive/
├── data/          ← ④ 画像ファイル群
└── annotations/   ← ③ train/val/test_coco_clean.json
```

---

## 編集が必要なファイル

画像・アノテーションの配置場所を変えた場合は以下を編集してください：

**`configs/dataset/phase11_detection.yml`**

```yaml
train_dataloader:
  dataset:
    img_folder: /datadrive/data/              ← 画像フォルダのパスに変更
    ann_file: /datadrive/annotations/train_coco_clean.json  ← JSONのパスに変更

val_dataloader:
  dataset:
    img_folder: /datadrive/data/              ← 同上
    ann_file: /datadrive/annotations/val_coco_clean.json    ← 同上
```

---

## アノテーションJSONの生成（新規データの場合）

画像ごとのJSONアノテーションからCOCO形式に変換します。

```
データ構造:
/your/data/
├── image001.jpg
├── image001.json   ← アノテーション
├── image002.jpg
├── image002.json
...
/your/cfg/
├── label.txt       ← クラス名（1行1クラス）
├── train.txt       ← 学習画像のファイル名一覧（拡張子なし）
├── val.txt
└── test.txt
```

```bash
# convert_to_coco.py の DATA_DIR / CFG_DIR / OUT_DIR を編集してから実行
python3 convert_to_coco.py
```

---

## 学習

```bash
torchrun --nproc_per_node=1 train.py \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11.yml
```

---

## 定量評価（クラス別 TP / FP / FN / mAP）

```bash
python3 tools/eval_per_class.py \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11.yml \
  -r outputs/deimv2_m_phase11_20260611/best_stg2.pth \
  -a /your/annotations/test_coco_clean.json \
  -i /your/data/ \
  -l /your/cfg/label.txt
```

---

## 定性評価（BBox可視化・推論速度・GPUメモリ計測）

```bash
python3 tools/inference/torch_inf_vis2.py \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11.yml \
  -r outputs/deimv2_m_phase11_20260611/best_stg2.pth \
  -d /your/data/ \
  -o /your/output/ \
  -l /your/cfg/label.txt \
  -a /your/annotations/test_coco_clean.json \
  --batch_size 1
```

---

## 動画推論

```bash
python3 tools/inference/torch_inf_video.py \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11.yml \
  -r outputs/deimv2_m_phase11_20260611/best_stg2.pth \
  -i /your/input.mp4 \
  -o /your/output.mp4 \
  -l /your/cfg/label.txt
```

---





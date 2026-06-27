"""
File: step24_benchmark_models.py

Purpose:
    Benchmark PyTorch segmentation checkpoints on the AI-Hub evaluation set.

Main Responsibilities:
    - Load AdvancedUNet and SMPHybridUNet checkpoints.
    - Compute dilated lane mIoU and inference FPS.
    - Render side-by-side input, ground truth, and model predictions.

Notes:
    Requires _dataset_aihub_eval/ and trained checkpoints. Outputs are written
    under benchmark_results/.
"""

import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.step11_advanced_model import AdvancedUNet
from src.models.step22_smp_model import SMPHybridUNet

from src import config as project_config

# ==========================================
# 1. 경로 및 하이퍼파라미터 설정
# ==========================================
DATASET_DIR = project_config.DATASET_AIHUB_EVAL_DIR
IMG_DIR = os.path.join(DATASET_DIR, "images")
MSK_DIR = os.path.join(DATASET_DIR, "masks")
OUTPUT_DIR = project_config.BENCHMARK_OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 4개의 모델 평가 라인업 구축
MODEL_WEIGHTS = {
    "Model A (Vanilla Pure)": project_config.ADVANCED_UNET_CHECKPOINT,
    "Model B (Vanilla Aug)": project_config.ADVANCED_AUG_UNET_CHECKPOINT,
    "Model C (ResNet34 Aug)": project_config.SMP_RESNET34_CHECKPOINT,
    "Model D (ResNet50 Aug)": project_config.SMP_RESNET50_CHECKPOINT,
}

TARGET_W, TARGET_H = 640, 360
CROP_Y = project_config.CROP_Y
DILATION_KERNEL = np.ones((15, 15), np.uint8) 
NUM_VISUALIZE = 5 

# ==========================================
# 2. 동적 모델 로더 함수
# ==========================================
def load_model(model_name, weight_path, device):
    if "ResNet50" in model_name:
        model = SMPHybridUNet(encoder_name="resnet50", classes=3).to(device)
    elif "ResNet34" in model_name:
        model = SMPHybridUNet(encoder_name="resnet34", classes=3).to(device)
    else:
        model = AdvancedUNet(in_channels=3, out_channels=3).to(device)
        
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device)['model_state_dict'])
        model.eval()
        return model
    else:
        print(f"⚠️ 경고: {weight_path} 가중치 파일을 찾을 수 없습니다. 평가를 건너뜁니다.")
        return None

# ==========================================
# 3. 메인 벤치마크 파이프라인
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 4-Tier 모델 벤치마크 엔진 가동 (Device: {device})")

    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])
    if not img_files:
        print("평가할 데이터가 없습니다.")
        return

    sample_indices = np.linspace(0, len(img_files) - 1, NUM_VISUALIZE, dtype=int)
    sample_files = [img_files[i] for i in sample_indices]
    
    vis_data = {f: {} for f in sample_files}
    metrics_report = {}

    for model_name, weight_path in MODEL_WEIGHTS.items():
        print(f"\n📊 [{model_name}] 평가 준비 중...")
        model = load_model(model_name, weight_path, device)
        if model is None: continue

        total_inter, total_union = 0, 0
        total_time = 0.0

        for img_name in tqdm(img_files, desc=f"Evaluating {model_name}"):
            img_path = os.path.join(IMG_DIR, img_name)
            msk_path = os.path.join(MSK_DIR, img_name.replace('.jpg', '.png'))

            image = cv2.imread(img_path)
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_cropped = frame_rgb[CROP_Y:, :, :].astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_cropped).permute(2, 0, 1).unsqueeze(0).to(device)

            gt_mask_full = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            gt_cropped = gt_mask_full[CROP_Y:, :]
            gt_dilated = cv2.dilate(gt_cropped, DILATION_KERNEL, iterations=1)
            gt_inds = (gt_dilated == 1) 

            if torch.cuda.is_available(): torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(img_tensor)
                pred_mask_cropped = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            total_time += (time.time() - start_time)

            pred_inds = (pred_mask_cropped == 1)
            total_inter += np.logical_and(pred_inds, gt_inds).sum()
            total_union += np.logical_or(pred_inds, gt_inds).sum()

            if img_name in sample_files:
                overlay = image.copy()
                pred_mask_full = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
                pred_mask_full[CROP_Y:, :] = pred_mask_cropped
                overlay[pred_mask_full == 1] = [0, 0, 255] 
                vis_data[img_name][model_name] = overlay
                
                if "GT" not in vis_data[img_name]:
                    gt_overlay = image.copy()
                    gt_overlay[gt_mask_full == 1] = [0, 255, 0] 
                    vis_data[img_name]["GT"] = gt_overlay
                    vis_data[img_name]["Input"] = image

        mIoU = (total_inter / total_union) * 100 if total_union > 0 else 0
        fps = len(img_files) / total_time if total_time > 0 else 0
        metrics_report[model_name] = {"mIoU": mIoU, "FPS": fps}

    # ==========================================
    # 4. 시각화 리포트 동적 생성 (Matplotlib)
    # ==========================================
    print("\n🎨 시각화 리포트 렌더링 중...")
    model_keys = list(metrics_report.keys())
    num_cols = 2 + len(model_keys) # Input + GT + Models
    
    fig, axes = plt.subplots(nrows=NUM_VISUALIZE, ncols=num_cols, figsize=(5 * num_cols, 15))
    col_titles = ["Input", "Ground Truth"] + model_keys

    for row, img_name in enumerate(sample_files):
        data = vis_data[img_name]
        
        images_to_plot = [
            cv2.cvtColor(data.get("Input", np.zeros_like(image)), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(data.get("GT", np.zeros_like(image)), cv2.COLOR_BGR2RGB)
        ]
        for key in model_keys:
            images_to_plot.append(cv2.cvtColor(data.get(key, np.zeros_like(image)), cv2.COLOR_BGR2RGB))

        for col, img_plot in enumerate(images_to_plot):
            axes[row, col].imshow(img_plot)
            axes[row, col].axis('off')
            if row == 0:
                # 폰트 크기 자동 조절 (모델 이름이 길어질 경우 대비)
                title_text = col_titles[col].split("(")[0].strip() if col > 1 else col_titles[col]
                axes[row, col].set_title(title_text, fontsize=14, fontweight='bold')

    plt.tight_layout()
    vis_path = os.path.join(OUTPUT_DIR, "benchmark_visualization_4tier.png")
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"✅ 시각화 이미지 저장 완료: {vis_path}")

    # ==========================================
    # 5. 콘솔 최종 리포트 출력
    # ==========================================
    print("\n" + "="*60)
    print("📊 [4-Tier Architecture Benchmark Report]")
    print("="*60)
    print(f"{'Model Name':<25} | {'Dilated mIoU (%)':<15} | {'Inference FPS':<10}")
    print("-" * 60)
    for name, metrics in metrics_report.items():
        print(f"{name:<25} | {metrics['mIoU']:<15.2f} | {metrics['FPS']:<10.1f}")
    print("="*60)

if __name__ == "__main__":
    main()

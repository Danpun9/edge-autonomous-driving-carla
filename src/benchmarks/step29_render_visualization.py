"""
File: step29_render_visualization.py

Purpose:
    Compose saved ONNX benchmark sample images into a comparison visualization.

Main Responsibilities:
    - Read input, ground-truth, and model overlay images from benchmark_results/samples/.
    - Render a compact comparison figure with Matplotlib.
    - Save the combined image under benchmark_results/.

Notes:
    Run after step28_benchmark_onnx.py has generated sample images.
"""

import os
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from src import config as project_config

# 백그라운드 렌더링 모드 설정 (GUI 창을 띄우지 않아 메모리 누수 방지)
matplotlib.use('Agg')

# ==========================================
# 1. 경로 및 모델 식별자 설정
# ==========================================
OUTPUT_DIR = project_config.BENCHMARK_OUTPUT_DIR
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")

# step28에서 저장된 파일명 식별자 매핑
MODEL_KEYS = [
    ("Input", "Input"),
    ("Ground Truth", "GT"),
    ("ResNet34 (FP32)", "ResNet34_FP32_Original"),
    ("ResNet34 (FP16)", "ResNet34_FP16_Half"),
    ("ResNet34 (INT8)", "ResNet34_INT8_Quantized")
]

def main():
    print("🎨 시각화 리포트 렌더링을 시작합니다...")
    
    if not os.path.exists(SAMPLE_DIR):
        print(f"⚠️ {SAMPLE_DIR} 폴더를 찾을 수 없습니다. 평가 스크립트를 먼저 실행해 주세요.")
        return

    # 샘플 폴더에서 원본 이미지(Input) 파일명 추출
    sample_files = sorted([f.split("_Input.jpg")[0] + ".jpg" for f in os.listdir(SAMPLE_DIR) if f.endswith("_Input.jpg")])
    
    if not sample_files:
        print("⚠️ 렌더링할 샘플 이미지가 없습니다.")
        return

    num_rows = len(sample_files)
    num_cols = len(MODEL_KEYS)
    
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 3 * num_rows))

    for row, base_img_name in enumerate(sample_files):
        for col, (title, suffix) in enumerate(MODEL_KEYS):
            img_path = os.path.join(SAMPLE_DIR, f"{base_img_name}_{suffix}.jpg")
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(img_rgb)
            else:
                # 이미지가 없을 경우 검은 화면 표시
                axes[row, col].imshow(np.zeros((360, 640, 3), dtype=np.uint8))
            
            axes[row, col].axis('off')
            
            # 첫 번째 행에만 타이틀 추가
            if row == 0:
                axes[row, col].set_title(title, fontsize=16, fontweight='bold')

    plt.tight_layout()
    vis_path = os.path.join(OUTPUT_DIR, "onnx_quantization_comparison.png")
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 시각화 렌더링 완료! 결과물 저장: {vis_path}")

if __name__ == "__main__":
    main()

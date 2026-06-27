"""
File: step13_advanced_inference_test.py

Purpose:
    Run sample inference for the multi-class segmentation checkpoint.

Main Responsibilities:
    - Load AdvancedUNet or SMPHybridUNet weights.
    - Preprocess images from the selected dataset directory.
    - Visualize class predictions against ground-truth masks.

Notes:
    The active DATA_DIR, CHECKPOINT_PATH, and model architecture must match the
    checkpoint that is being inspected.
"""

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from src import config as project_config

# 이전 단계에서 만든 다중 클래스 모델 임포트
from src.models.step11_advanced_model import AdvancedUNet
from src.models.step22_smp_model import SMPHybridUNet

# ==========================================
# 1. 하이퍼파라미터 및 설정
# ==========================================
# DATA_DIR = "_dataset_multiclass"
DATA_DIR = project_config.DATASET_AUG_UNET_DIR
IMG_DIR = os.path.join(DATA_DIR, "images")
# CHECKPOINT_PATH = "advanced_best_aug_unet_model.pth" # AdvancedUNet용
CHECKPOINT_PATH = project_config.SMP_RESNET50_CHECKPOINT
CROP_Y = project_config.CROP_Y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 다중 클래스 추론 디바이스: {device}")

# ==========================================
# 2. 전처리 함수
# ==========================================
def preprocess_image(img_path, crop_y):
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 상단 50% 크롭
    image_cropped = image_rgb[crop_y:, :, :]
    
    # 정규화 및 텐서 변환 (Float32)
    image_norm = image_cropped.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor.to(device), image_cropped

# ==========================================
# 3. 메인 추론 및 시각화 테스트
# ==========================================
def main():
    # 1. 모델 초기화 및 학습된 가중치 로드
    # model = AdvancedUNet(in_channels=3, out_channels=3).to(device)
    model = SMPHybridUNet(encoder_name="resnet50", classes=3).to(device)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {CHECKPOINT_PATH}")
        return

    print(f"💾 체크포인트 로드 중: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # 추론 모드 전환 (매우 중요)
    print("✅ 모델 추론 준비 완료!")

    # 2. 테스트할 이미지 무작위 선택 (5장)
    img_names = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
    num_samples = min(5, len(img_names))
    selected_imgs = random.sample(img_names, num_samples)

    # 시각화 설정
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3 * num_samples))
    fig.suptitle("Advanced U-Net: [Input] vs [Argmax Class] vs [Overlay]", fontsize=16)

    with torch.no_grad(): # 기울기 계산 비활성화
        for i, img_name in enumerate(selected_imgs):
            img_path = os.path.join(IMG_DIR, img_name)
            
            # 전처리
            image_tensor, image_cropped = preprocess_image(img_path, CROP_Y)
            
            # 3. 모델 추론 (출력 차원: [1, 3, 180, 640])
            outputs = model(image_tensor)
            
            # [핵심] Argmax를 통한 클래스 결정 (차원: [1, 180, 640])
            # 3개의 채널 중 가장 값이 큰 채널의 '인덱스(0, 1, 2)'를 픽셀 값으로 취합니다.
            pred_mask = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()

            # 4. 시각화용 컬러 레이어 생성
            color_mask = np.zeros_like(image_cropped)
            
            # Class 1 (차선) -> 빨간색 [255, 0, 0]
            color_mask[pred_mask == 1] = [255, 0, 0]
            
            # Class 2 (횡단보도/격자) -> 초록색 [0, 255, 0]
            color_mask[pred_mask == 2] = [0, 255, 0]
            
            # 5. 원본 이미지 위에 반투명(alpha=0.5) 오버레이
            overlay = cv2.addWeighted(image_cropped, 0.6, color_mask, 0.4, 0)

            # 결과 출력
            axes[i, 0].imshow(image_cropped)
            axes[i, 0].set_title(f"Input: {img_name}")
            axes[i, 0].axis('off')
            
            # 예측 마스크 (0, 1, 2 값을 보기 좋게 시각화)
            axes[i, 1].imshow(pred_mask, cmap='viridis', vmin=0, vmax=2)
            axes[i, 1].set_title("Argmax Class Map (0,1,2)")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title("Overlay (Red:Lane, Green:Crosswalk)")
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

if __name__ == "__main__":
    main()

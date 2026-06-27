"""
File: step13_inference_test.py

Purpose:
    Run sample inference for the binary U-Net checkpoint.

Main Responsibilities:
    - Load best_unet_model.pth.
    - Preprocess images from _dataset/.
    - Visualize input, ground truth mask, and predicted binary mask.

Notes:
    This is a local inspection script and expects the trained checkpoint and
    dataset to exist.
"""

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from src import config as project_config

# 이전 단계에서 만든 모듈 임포트
from src.models.step11_model import UNet

# ==========================================
# 1. 하이퍼파라미터 및 설정
# ==========================================
DATA_DIR = project_config.DATASET_BINARY_DIR
IMG_DIR = os.path.join(DATA_DIR, "images")
CHECKPOINT_PATH = project_config.BINARY_UNET_CHECKPOINT
CROP_Y = project_config.CROP_Y
THRESHOLD = 0.5        # 차선 판별 임계값

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 추론 디바이스: {device}")

# ==========================================
# 2. 전처리 함수 (학습 때와 동일해야 함)
# ==========================================
def preprocess_image(img_path, crop_y):
    # OpenCV로 이미지 로드 (BGR -> RGB 변환)
    image = cv2.imread(img_path)
    image_bgr = np.copy(image) # 시각화용 BGR 원본 보관
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1. ROI Cropping (상단 50% 잘라내기)
    image_cropped = image[crop_y:, :, :]
    
    # 2. 정규화 (0.0 ~ 1.0)
    image_norm = image_cropped.astype(np.float32) / 255.0
    
    # 3. PyTorch Tensor 변환 (HWC -> CHW, 배치 차원 추가)
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor.to(device), image_cropped

# ==========================================
# 3. 메인 추론 및 시각화 테스트
# ==========================================
def main():
    # 1. 모델 초기화 및 가중치 로드
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {CHECKPOINT_PATH}")
        print("step12_train.py를 먼저 실행하여 모델을 학습시켜 주세요.")
        return

    print(f"💾 체크포인트 로드 중: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # [핵심] 모델을 추론 모드로 전환 (BatchNorm, Dropout 동작 변경)
    model.eval()
    print("✅ 모델 추론 준비 완료!")

    # 2. 테스트할 임의의 이미지 선택
    img_names = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
    if not img_names:
        print(f"❌ 이미지를 찾을 수 없습니다: {IMG_DIR}")
        return
    
    # 테스트하고 싶은 이미지 수를 조절 (예: 5장)
    num_samples = min(5, len(img_names))
    selected_imgs = random.sample(img_names, num_samples)

    # 시각화 설정
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3 * num_samples))
    fig.suptitle("U-Net Lane Detection Inference: [Input] vs [Pred Mask] vs [Overlay]", fontsize=16)

    with torch.no_grad(): # [핵심] 기울기 계산 비활성화 (메모리 절약 및 속도 향상)
        for i, img_name in enumerate(selected_imgs):
            img_path = os.path.join(IMG_DIR, img_name)
            
            # 전처리
            image_tensor, image_cropped = preprocess_image(img_path, CROP_Y)
            
            # 3. 모델 추론
            output = model(image_tensor)
            
            # 로릿(Logit) 값을 시그모이드를 통해 확률(0.0~1.0)로 변환
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            
            # 임계값(Threshold) 적용하여 이진화
            pred_mask = np.where(prob >= THRESHOLD, 1.0, 0.0)

            # 4. 시각화용 Overlay 생성
            h, w = pred_mask.shape
            overlay = np.copy(image_cropped)
            
            # 차선으로 예측된 영역(pred_mask == 1)을 초록색으로 색칠
            green_layer = np.zeros_like(image_cropped)
            green_layer[pred_mask == 1] = [0, 255, 0] # RGB에서 Green
            
            # 원본 이미지 위에 초록색 레이어를 투명도(alpha=0.5)를 주어 덧씌움
            alpha = 0.5
            cv2.addWeighted(green_layer, alpha, overlay, 1 - alpha, 0, overlay)

            # 결과 출력
            # [Input RGB]
            axes[i, 0].imshow(image_cropped)
            axes[i, 0].set_title(f"Input: {img_name}")
            axes[i, 0].axis('off')
            
            # [Predicted Mask]
            axes[i, 1].imshow(pred_mask, cmap='gray')
            axes[i, 1].set_title("Predicted Mask")
            axes[i, 1].axis('off')
            
            # [Overlay Result]
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title("Overlay Result")
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # 제목 공간 확보
    plt.show()

if __name__ == "__main__":
    main()

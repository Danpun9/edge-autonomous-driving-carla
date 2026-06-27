"""
File: step10_dataset.py

Purpose:
    Define a PyTorch Dataset for binary CARLA lane segmentation.

Main Responsibilities:
    - Load image/mask pairs from _dataset/.
    - Crop the top portion of frames to focus on the road region.
    - Normalize RGB inputs and convert lane masks to binary tensors.

Related Files:
    - step12_train.py trains the binary U-Net with this dataset.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from src import config as project_config

# ==========================================
# 1. PyTorch 커스텀 데이터셋 클래스 정의
# ==========================================
class CarlaLaneDataset(Dataset):
    def __init__(self, data_dir=project_config.DATASET_BINARY_DIR, crop_y=project_config.CROP_Y):
        """
        Args:
            data_dir (str): 데이터셋이 저장된 최상위 경로
            crop_y (int): 이미지 상단에서 잘라낼 픽셀 수 (기본값 180, 즉 상단 50% 크롭)
        """
        self.img_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        
        # 디렉토리 내의 모든 파일 이름을 리스트로 정렬하여 저장
        self.img_names = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
        self.crop_y = crop_y

    def __len__(self):
        # 전체 데이터의 개수 반환
        return len(self.img_names)

    def __getitem__(self, idx):
        # 1. 파일 경로 매칭 및 읽기
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # OpenCV로 이미지 로드 (BGR -> RGB 변환 필수!)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 정답지 마스크는 흑백(Grayscale)으로 로드
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # ==========================================
        # 2. ROI Cropping (상단 50% 잘라내기)
        # ==========================================
        # 하늘, 건물 등 차선 인식에 불필요한 영역을 날려버려 연산량 50% 절감
        image = image[self.crop_y:, :, :]
        mask = mask[self.crop_y:, :]

        # ==========================================
        # 3. 정규화 (Normalization) 및 이진화 (Binarization)
        # ==========================================
        # 픽셀값(0~255)을 딥러닝 모델이 좋아하는 (0.0 ~ 1.0) 사이의 Float 타입으로 변환
        image = image.astype(np.float32) / 255.0
        
        # 마스크에서 차선이 있는 곳은 1.0, 배경은 0.0으로 완벽한 이진분류 정답지 생성
        mask = np.where(mask > 0, 1.0, 0.0).astype(np.float32)

        # ==========================================
        # 4. PyTorch Tensor 변환
        # ==========================================
        # HWC(Height, Width, Channel) -> CHW(Channel, Height, Width) 포맷 변경
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        
        # 마스크는 채널 차원이 없으므로 강제로 [1, H, W] 차원을 만들어줌
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor


# ==========================================
# [테스트 블록] 데이터 로더가 잘 작동하는지 시각화
# ==========================================
if __name__ == "__main__":
    print("=== Step 10: PyTorch Dataset Pipeline Validation ===")
    
    # 데이터셋 인스턴스 생성
    dataset = CarlaLaneDataset(data_dir=project_config.DATASET_BINARY_DIR, crop_y=project_config.CROP_Y)
    print(f"총 {len(dataset)}장의 데이터가 로드되었습니다.")

    # DataLoader 생성 (배치 사이즈 4로 설정하여 GPU에 한 번에 4장씩 올릴 준비)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # 첫 번째 배치를 뽑아와서 확인
    images, masks = next(iter(dataloader))
    
    print(f"입력 이미지 텐서 형태 (B, C, H, W): {images.shape}")
    print(f"정답지 마스크 텐서 형태 (B, C, H, W): {masks.shape}")

    # 시각화 검증 (Tensor를 다시 Numpy 이미지로 변환하여 출력)
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    fig.suptitle("Cropped Tensors: [Input RGB] vs [Ground Truth Mask]", fontsize=16)

    for i in range(4):
        # Tensor(C, H, W) -> Numpy(H, W, C) 변환
        img_display = images[i].permute(1, 2, 0).numpy()
        mask_display = masks[i].squeeze().numpy()

        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f"Input {i+1}")
        axes[0, i].axis('off')

        axes[1, i].imshow(mask_display, cmap='gray')
        axes[1, i].set_title(f"Mask {i+1}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

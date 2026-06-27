"""
File: step10_advanced_dataset.py

Purpose:
    Define a PyTorch Dataset for multi-class CARLA road-marking segmentation.

Main Responsibilities:
    - Load image/mask pairs from _dataset_multiclass or augmented datasets.
    - Crop the road ROI and normalize RGB inputs.
    - Return class-index masks for CrossEntropyLoss.

Related Files:
    - step12_advanced_train.py trains AdvancedUNet or SMPHybridUNet with this
      dataset.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from src import config as project_config

# ==========================================
# 1. 다중 클래스 지원 커스텀 데이터셋 클래스
# ==========================================
class AdvancedCarlaDataset(Dataset):
    def __init__(self, data_dir=project_config.DATASET_MULTICLASS_DIR, crop_y=project_config.CROP_Y):
        """
        Args:
            data_dir (str): 다중 클래스 데이터셋이 저장된 경로
            crop_y (int): 이미지 상단에서 잘라낼 픽셀 수 (연산량 50% 절감 및 하늘 노이즈 제거)
        """
        self.img_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        
        self.img_names = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
        self.crop_y = crop_y

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # 1. 원본 이미지 로드 (RGB 변환)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 정답지 로드 (Grayscale, 0, 1, 2 값 유지)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # ==========================================
        # [핵심 1] ROI Cropping (상단 50% 잘라내기)
        # ==========================================
        image = image[self.crop_y:, :, :]
        mask = mask[self.crop_y:, :]

        # ==========================================
        # [핵심 2] 텐서 변환의 분리 (RGB vs Mask)
        # ==========================================
        # 입력 이미지(X): 딥러닝이 소화할 수 있게 0.0~1.0 사이의 Float32로 정규화
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1) # [C, H, W]
        
        # 정답지 마스크(Y): 다중 클래스 분류를 위해 절대 실수로 변환하거나 255로 나누지 않음!
        # PyTorch의 CrossEntropyLoss는 [H, W] 형태의 Long(정수) 텐서를 요구합니다.
        # 이전처럼 unsqueeze(0)로 채널을 추가하지 않는 것이 매우 중요합니다.
        mask_tensor = torch.from_numpy(mask).long() # [H, W] 형태 유지

        return image_tensor, mask_tensor

# ==========================================
# [테스트 블록] 텐서 차원 검증 및 데이터 시각화
# ==========================================
if __name__ == "__main__":
    print("=== Step 10: Advanced Multi-class Dataset Validation & Visualization ===")
    dataset = AdvancedCarlaDataset(data_dir=project_config.DATASET_MULTICLASS_DIR, crop_y=project_config.CROP_Y)
    
    if len(dataset) == 0:
        print("데이터셋 경로 '_dataset_multiclass'에 이미지가 존재하는지 확인해주세요.")
    else:
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        images, masks = next(iter(dataloader))
        
        print(f"입력 이미지 텐서 형태 (B, C, H, W): {images.shape}")
        print(f"정답지 마스크 텐서 형태 (B, H, W):    {masks.shape} -> 채널(C)이 없는 것이 정상입니다!")
        print(f"마스크 내에 존재하는 클래스 종류:      {torch.unique(masks).tolist()} -> [0, 1, 2]가 나와야 합니다.")
        print("✅ 데이터 로더 무결성 검증 완료!")

        # 데이터 시각화 (배치 내 이미지)
        batch_size = images.shape[0]
        fig, axes = plt.subplots(batch_size, 2, figsize=(10, 3 * batch_size))
        
        if batch_size == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i in range(batch_size):
            # 1. 이미지 시각화를 위해 텐서를 NumPy 배열로 변환 [C, H, W] -> [H, W, C]
            img_np = images[i].permute(1, 2, 0).numpy()
            
            # 2. 마스크 시각화를 위해 텐서를 NumPy 배열로 변환 [H, W]
            mask_np = masks[i].numpy()
            
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Sample {i+1} - RGB Image")
            axes[i, 0].axis('off')
            
            # 마스크는 컬러맵을 적용하여 시각화 (0, 1, 2 범위를 명확히)
            axes[i, 1].imshow(mask_np, cmap='viridis', vmin=0, vmax=2)
            axes[i, 1].set_title(f"Sample {i+1} - Mask (Class 0, 1, 2)")
            axes[i, 1].axis('off')
            
        plt.tight_layout()
        plt.show()

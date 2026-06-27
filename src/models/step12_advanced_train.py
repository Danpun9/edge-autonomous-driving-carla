"""
File: step12_advanced_train.py

Purpose:
    Train the multi-class segmentation model used for Sim2Real experiments.

Main Responsibilities:
    - Load AdvancedCarlaDataset from the augmented U-Net dataset.
    - Train AdvancedUNet or SMPHybridUNet with weighted CrossEntropyLoss.
    - Save and resume checkpoints for the selected architecture.

Notes:
    The active model and checkpoint path are controlled by constants near the
    top of the file. Keep them aligned with downstream export/benchmark scripts.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src import config as project_config

# 방금 전 단계에서 만든 Advanced 모듈 임포트
from src.models.step10_advanced_dataset import AdvancedCarlaDataset
from src.models.step11_advanced_model import AdvancedUNet
from src.models.step22_smp_model import SMPHybridUNet

# ==========================================
# 1. 하이퍼파라미터 및 설정
# ==========================================
BATCH_SIZE = project_config.BATCH_SIZE
EPOCHS = project_config.EPOCHS
LEARNING_RATE = project_config.LEARNING_RATE
# CHECKPOINT_PATH = "advanced_best_aug_unet_model.pth" # AdvancedUNet용
CHECKPOINT_PATH = project_config.SMP_RESNET34_CHECKPOINT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Advanced 다중 클래스 학습 디바이스: {device}")

def main():
    # ==========================================
    # 2. 데이터셋 로드 및 분할 (8:2)
    # ==========================================
    # dataset = AdvancedCarlaDataset(data_dir="_dataset_multiclass", crop_y=180)
    dataset = AdvancedCarlaDataset(
        data_dir=project_config.DATASET_AUG_UNET_DIR,
        crop_y=project_config.CROP_Y,
    )
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"📊 데이터셋 준비 완료: Train {train_size}장, Validation {val_size}장")

    # ==========================================
    # 3. 모델 및 옵티마이저 초기화
    # ==========================================
    # model = AdvancedUNet(in_channels=3, out_channels=3).to(device)
    model = SMPHybridUNet(encoder_name="resnet34", classes=3).to(device) # resnet 인코더 모델

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================================
    # 4. [핵심] 다중 클래스 손실 함수 정의 (Weighted Cross-Entropy)
    # ==========================================
    # 도로 이미지의 90%가 배경(0), 차선(1)은 2%, 횡단보도(2)는 8%를 차지합니다.
    # 가중치를 주지 않으면 모델은 모든 픽셀을 배경(0)으로 찍는 '꼼수'를 부립니다.
    # [배경 가중치, 차선 가중치, 횡단보도 가중치]
    weights = torch.tensor([0.1, 5.0, 2.0], dtype=torch.float32).to(device)
    
    # CrossEntropyLoss는 내부적으로 LogSoftmax와 NLLLoss를 결합하여 매우 안정적으로 연산됩니다.
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ==========================================
    # 5. 체크포인트 Resume 로직 (결함 수용)
    # ==========================================
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(CHECKPOINT_PATH):
        print(f"💾 기존 체크포인트 발견! [{CHECKPOINT_PATH}] 복구를 시작합니다...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            print(f"✅ 복구 완료: 에포크 {start_epoch}부터 이어서 학습합니다. (최고 Loss: {best_val_loss:.4f})")
        except RuntimeError as e:
            print("\n❌ [오류] 기존 체크포인트의 모델 아키텍처와 현재 모델 아키텍처가 일치하지 않습니다.")
            print(f"   현재 사용 중인 모델: {model.__class__.__name__}")
            print(f"   체크포인트 파일 경로: {CHECKPOINT_PATH}")
            print("   해결책: 다른 모델 아키텍처를 학습할 때는 CHECKPOINT_PATH 파일명을 변경하거나 기존 파일을 다른 곳으로 이동하십시오.\n")
            raise e
    else:
        print("🌱 저장된 체크포인트가 없습니다. 처음부터 학습을 시작합니다.")

    # ==========================================
    # 6. 메인 학습 루프
    # ==========================================
    for epoch in range(start_epoch, EPOCHS):
        # ------------------------------------
        # [훈련 단계] Train Phase
        # ------------------------------------
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS-1} [Train]")
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            # images: [B, 3, 180, 640] (Float)
            # masks:  [B, 180, 640] (Long) - 정수형 클래스 인덱스

            optimizer.zero_grad()
            
            # 순전파: outputs 형태는 [B, 3, 180, 640] (Logits)
            outputs = model(images)
            
            # 오차 계산: CrossEntropyLoss가 [B, 3, H, W]와 [B, H, W]를 알아서 매칭하여 계산
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # ------------------------------------
        # [검증 단계] Validation Phase
        # ------------------------------------
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS-1} [Valid]"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"📈 Epoch {epoch} 결과: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ------------------------------------
        # [저장 단계] Checkpoint Save
        # ------------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"✨ 새로운 최고 성능 달성! 다중 클래스 모델을 저장합니다. (Val Loss: {best_val_loss:.4f})")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, CHECKPOINT_PATH)
            
    print("🎉 Advanced 다중 클래스 학습이 성공적으로 종료되었습니다!")

if __name__ == "__main__":
    main()

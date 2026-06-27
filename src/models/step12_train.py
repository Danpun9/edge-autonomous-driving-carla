"""
File: step12_train.py

Purpose:
    Train the binary U-Net lane segmentation model.

Main Responsibilities:
    - Load CarlaLaneDataset from _dataset/.
    - Train UNet with a combined BCE and Dice loss.
    - Save and resume checkpoints at best_unet_model.pth.

Notes:
    Requires prepared dataset files and PyTorch. Training writes model
    checkpoints, which are intentionally excluded from Git.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src import config as project_config

# 이전 단계에서 만든 모듈 임포트
from src.models.step10_dataset import CarlaLaneDataset
from src.models.step11_model import UNet

# ==========================================
# 1. 하이퍼파라미터 및 설정
# ==========================================
BATCH_SIZE = project_config.BATCH_SIZE       # GPU 메모리에 맞게 조절
EPOCHS = project_config.EPOCHS               # 총 학습 세대 수
LEARNING_RATE = project_config.LEARNING_RATE # Adam 옵티마이저 학습률
CHECKPOINT_PATH = project_config.BINARY_UNET_CHECKPOINT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 학습 디바이스: {device}")

# ==========================================
# 2. 손실 함수 (Loss Function) 정의: BCE + Dice Loss
# ==========================================
class DiceBCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCEWithLogitsLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        # 1. BCE Loss 계산 (내부적으로 Sigmoid 적용됨)
        bce_loss = self.bce(inputs, targets)
        
        # 2. Dice Loss 계산을 위해 Sigmoid 적용
        inputs = torch.sigmoid(inputs)
        
        # 텐서를 1차원으로 펼침 (Flatten)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth))
        
        # 3. 두 Loss를 5:5 비율로 융합
        return 0.5 * bce_loss + 0.5 * dice_loss

# ==========================================
# 3. 메인 학습 루프
# ==========================================
def main():
    # 1. 데이터셋 로드 및 Train/Validation 분할 (8:2 비율)
    dataset = CarlaLaneDataset(
        data_dir=project_config.DATASET_BINARY_DIR,
        crop_y=project_config.CROP_Y,
    )
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    
    # 랜덤 시드를 고정하지 않아 매번 다양하게 분할되도록 함
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"📊 데이터셋 준비 완료: Train {train_size}장, Validation {val_size}장")

    # 2. 모델, 손실 함수, 옵티마이저 초기화
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = DiceBCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. [핵심] 체크포인트 Resume 로직
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(CHECKPOINT_PATH):
        print(f"💾 기존 체크포인트 발견! [{CHECKPOINT_PATH}] 복구를 시작합니다...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
        # 뇌의 상태(가중치)와 발자취(옵티마이저) 복원
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        
        print(f"✅ 복구 완료: 에포크 {start_epoch}부터 이어서 학습합니다. (이전 최고 Loss: {best_val_loss:.4f})")
    else:
        print("🌱 저장된 체크포인트가 없습니다. 처음부터 학습을 시작합니다.")

    # 4. 에포크(Epoch) 순회
    for epoch in range(start_epoch, EPOCHS):
        # ------------------------------------
        # [훈련 단계] Train Phase
        # ------------------------------------
        model.train()
        train_loss = 0.0
        
        # tqdm으로 진행률 바 표시
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS-1} [Train]")
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)

            # 기울기 초기화
            optimizer.zero_grad()
            
            # 순전파 (추론)
            outputs = model(images)
            
            # 오차 계산 및 역전파
            loss = criterion(outputs, masks)
            loss.backward()
            
            # 가중치 업데이트
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # ------------------------------------
        # [검증 단계] Validation Phase
        # ------------------------------------
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad(): # 검증 시에는 기울기를 계산하지 않음 (메모리 절약)
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
        # 검증 오차가 이전 최고 기록보다 낮아졌을 때만 뇌를 저장함
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"✨ 새로운 최고 성능 달성! 모델을 저장합니다. (Val Loss: {best_val_loss:.4f})")
            
            # 모델 가중치, 옵티마이저 상태, 에포크, 최고 점수를 모두 딕셔너리로 포장하여 저장
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, CHECKPOINT_PATH)
            
    print("🎉 모든 학습이 성공적으로 종료되었습니다!")

if __name__ == "__main__":
    main()

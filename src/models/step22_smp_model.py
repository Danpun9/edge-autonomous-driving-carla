"""
File: step22_smp_model.py

Purpose:
    Wrap segmentation_models_pytorch U-Net backbones for multi-class road
    segmentation.

Main Responsibilities:
    - Build a U-Net with a configurable encoder such as ResNet34 or ResNet50.
    - Return raw logits for background, lane, and crosswalk/marking classes.
    - Provide a small tensor-shape validation when run directly.

Notes:
    Downstream export and benchmark scripts depend on this class name and output
    shape.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SMPHybridUNet(nn.Module):
    """
    사전 학습된 ResNet-50 백본을 사용하는 차선/횡단보도 분할 하이브리드 모델
    """
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=3):
        super(SMPHybridUNet, self).__init__()
        
        # SMP 라이브러리를 이용한 U-Net 구축
        self.model = smp.Unet(
            encoder_name=encoder_name,        # 백본 네트워크 (resnet34, resnet50, efficientnet 등)
            encoder_weights=encoder_weights,  # ImageNet 사전 학습 가중치 사용
            in_channels=in_channels,          # 입력 채널 (RGB: 3)
            classes=classes,                  # 출력 클래스 (0: 배경, 1: 차선, 2: 횡단보도)
            activation=None                   # CrossEntropyLoss와 함께 사용하기 위해 raw logits 출력
        )

    def forward(self, x):
        # x.shape = (Batch, 3, H, W)
        return self.model(x)

# 모델 정상 작동 테스트용 코드
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"테스트 디바이스: {device}")
    
    # 3채널 입력, 3채널 출력을 가지는 ResNet-50 U-Net 생성
    model = SMPHybridUNet().to(device)
    
    # 더미 데이터 생성 (배치1, 3채널, 180높이, 640너비)
    dummy_input = torch.randn(1, 3, 180, 640).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"입력 텐서 형태: {dummy_input.shape}")
    print(f"출력 텐서 형태: {output.shape} (기대값: [1, 3, 180, 640])")
    print("✅ SMP U-Net 모델이 성공적으로 생성되었습니다!")

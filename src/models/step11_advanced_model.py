"""
File: step11_advanced_model.py

Purpose:
    Define a multi-class U-Net for lane and crosswalk segmentation.

Main Responsibilities:
    - Use the same basic U-Net structure as step11_model.py.
    - Produce three raw-logit output channels for class-index masks.
    - Validate expected tensor shapes when run directly.

Notes:
    Outputs are logits for CrossEntropyLoss. Apply softmax/argmax outside the
    model during inference or visualization.
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Convolution => Batch Normalization => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class AdvancedUNet(nn.Module):
    # [핵심] out_channels 기본값을 1에서 3으로 변경 (0: 배경, 1: 차선, 2: 횡단보도)
    def __init__(self, in_channels=3, out_channels=3, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 인코더
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 병목 계층
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # 디코더
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # 최종 출력 레이어 (3개의 채널을 뱉어냅니다)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]

            if x.shape != skip_connection.shape:
                import torchvision.transforms.functional as TF
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip)

        # ※ 다중 클래스에서도 최종단에 Softmax를 씌우지 않고 순수 Logit을 반환합니다.
        # nn.CrossEntropyLoss 내부에서 LogSoftmax가 최적화되어 돌아가기 때문입니다.
        return self.final_conv(x)

# ==========================================
# [테스트 블록] 모델 차원 무결성 검증
# ==========================================
if __name__ == "__main__":
    print("=== Step 11: Advanced U-Net Validation ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 더미 입력 텐서 (크롭된 해상도 180x640 적용)
    x = torch.randn((4, 3, 180, 640)).to(device)
    model = AdvancedUNet(in_channels=3, out_channels=3).to(device)
    
    preds = model(x)

    print(f"입력 텐서 형태: {x.shape}")
    print(f"출력 텐서 형태: {preds.shape}")
    
    # 출력이 3채널인지 엄격하게 검사
    assert preds.shape == (4, 3, 180, 640), "출력 채널이 3개가 아닙니다!"
    print("✅ 3채널 출력 U-Net 아키텍처 무결성 검증 완료!")

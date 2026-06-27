"""
File: step11_model.py

Purpose:
    Define the lightweight binary U-Net used for early lane segmentation.

Main Responsibilities:
    - Implement DoubleConv blocks and encoder/decoder skip connections.
    - Return raw logits for one output channel.
    - Provide a small shape-validation block when run directly.

Notes:
    Do not add a Sigmoid layer inside the model; step12_train.py uses
    BCEWithLogitsLoss-compatible outputs.
"""

import torch
import torch.nn as nn

# ==========================================
# 1. 반복 사용되는 기본 블록 (Conv -> BatchNorm -> ReLU)
# ==========================================
class DoubleConv(nn.Module):
    """(Convolution => Batch Normalization => ReLU)를 두 번 반복하는 블록"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # 커널 사이즈 3, 패딩 1을 주어 이미지 해상도가 줄어들지 않도록 유지합니다.
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# ==========================================
# 2. U-Net 전체 아키텍처
# ==========================================
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        """
        Args:
            in_channels: 입력 채널 수 (RGB = 3)
            out_channels: 출력 채널 수 (차선 유무 이진 분류 = 1)
            features: 각 계층(Layer)의 필터/채널 개수 (경량화를 위해 32부터 시작)
        """
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ------------------------------------
        # 인코더 (Encoder - 수축 경로)
        # ------------------------------------
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # ------------------------------------
        # 병목 계층 (Bottleneck)
        # ------------------------------------
        # 이미지의 크기가 가장 작고, 추상적인 문맥(Context) 정보가 가장 응축된 곳입니다.
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # ------------------------------------
        # 디코더 (Decoder - 확장 경로)
        # ------------------------------------
        for feature in reversed(features):
            # Transposed Convolution (업샘플링: 해상도를 2배로 키움)
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            # Skip Connection 결합 후 특징을 정제하는 Conv 블록
            self.ups.append(DoubleConv(feature*2, feature))

        # ------------------------------------
        # 최종 출력 계층 (Output Layer)
        # ------------------------------------
        # 최종적으로 1개의 채널(흑백 마스크)로 만들어주는 1x1 Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # 1. 수축 (Downsampling)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x) # 디코더에 넘겨주기 위해 현재 해상도의 특징 맵을 저장
            x = self.pool(x)

        # 2. 병목 (Bottleneck)
        x = self.bottleneck(x)
        
        # 스킵 커넥션 리스트를 역순으로 뒤집어 디코더와 순서를 맞춤
        skip_connections = skip_connections[::-1]

        # 3. 확장 (Upsampling) 및 스킵 커넥션 (Skip Connection) 결합
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x) # 해상도 2배 증가
            skip_connection = skip_connections[i//2] # 저장해둔 고해상도 특징 맵 꺼내기

            # [방어 코드] MaxPool 과정에서 픽셀이 홀수여서 잘린 경우, 해상도를 강제로 맞춰줌
            if x.shape != skip_connection.shape:
                import torchvision.transforms.functional as TF
                x = TF.resize(x, size=skip_connection.shape[2:])

            # 핵심 마법: 위치 정보(skip_connection)와 추상 정보(x)를 채널 차원(dim=1)에서 합체
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip)

        # ※ 주의: 최종 출력에 Sigmoid()를 씌우지 않습니다. 
        # 다음 단계에서 사용할 BCEWithLogitsLoss가 내부적으로 계산해 주는 것이 수학적으로 훨씬 안정적이기 때문입니다.
        return self.final_conv(x)

# ==========================================
# [테스트 블록] 모델 차원 무결성 검증
# ==========================================
if __name__ == "__main__":
    print("=== Step 11: U-Net Architecture Validation ===")
    
    # 1. RTX 4080 Super 환경을 가정한 더미 데이터 텐서 생성 (배치 4, 채널 3, 높이 180, 너비 640)
    # GPU가 있다면 GPU로, 없다면 CPU로 올림
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((4, 3, 180, 640)).to(device)
    
    # 2. 모델 인스턴스화
    model = UNet(in_channels=3, out_channels=1).to(device)

    # 3. 네트워크 통과 (Inference)
    preds = model(x)

    print(f"사용 디바이스: {device}")
    print(f"입력 텐서 형태 (Input X):  {x.shape}")
    print(f"출력 텐서 형태 (Output Y): {preds.shape}")
    
    # 출력 형태가 입력의 해상도(180x640)와 동일하고, 채널이 1인지 검증
    assert preds.shape == (4, 1, 180, 640), "출력 텐서의 차원이 파괴되었습니다!"
    print("✅ 모델 차원 무결성 검증 통과! (네트워크 구조가 완벽합니다)")

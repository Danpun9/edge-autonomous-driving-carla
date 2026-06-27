"""
File: visualize_unet.py

Purpose:
    Generate a visual graph of the AdvancedUNet architecture.

Main Responsibilities:
    - Instantiate AdvancedUNet.
    - Use torchview/Graphviz-style tooling to render the model structure.
    - Save the architecture visualization artifact.

Notes:
    Requires optional visualization dependencies such as torchview and graphviz.
"""

import torch
from src.models.step11_advanced_model import AdvancedUNet

# 필요한 라이브러리 자동 설치 안내 및 임포트 시도
try:
    from torchinfo import summary
except ImportError:
    print("torchinfo 라이브러리가 설치되어 있지 않습니다. 설치를 시도합니다...")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchinfo"])
    from torchinfo import summary

try:
    from torchview import draw_graph
except ImportError:
    print(
        "torchview 및 graphviz 라이브러리가 설치되어 있지 않습니다. 설치를 시도합니다..."
    )
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "torchview graphviz"]
    )
    from torchview import draw_graph


def main():
    # 1. 모델 생성
    model = AdvancedUNet(in_channels=3, out_channels=3)

    # 입력 크기 설정 (step11에 정의된 크롭된 해상도 180x640 적용)
    input_size = (1, 3, 180, 640)

    print("\n" + "=" * 50)
    print(" 1. U-Net 모델 요약 (Summary)")
    print("=" * 50)

    # 2. torchinfo를 사용하여 상세 요약 정보 출력
    summary(
        model,
        input_size=input_size,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds",
        ],
        depth=3,
    )

    print("\n" + "=" * 50)
    print(" 2. U-Net 연결 구조 시각화 (Graph Generation)")
    print("=" * 50)
    print("모델의 구조 그래프를 그립니다...")

    try:
        # 3. torchview를 사용하여 연산 그래프 시각화 및 이미지 파일로 저장
        # (Graphviz가 시스템에 설치되어 있어야 이미지로 변환이 가능합니다)
        model_graph = draw_graph(
            model,
            input_size=input_size,
            expand_nested=True,
            depth=3,
            device="cpu",
            save_graph=True,
            filename="unet_architecture",
        )
        print(
            "✅ 시각화 완료! 'unet_architecture.png' (또는 pdf/gv) 파일이 생성되었습니다."
        )
    except Exception as e:
        print(f"❌ Graphviz 이미지 생성 실패: {e}")
        print(
            "Graphviz 소프트웨어가 시스템에 설치되어 있지 않거나 환경 변수(PATH)에 등록되어 있지 않을 수 있습니다."
        )
        print(
            "도움말: Windows의 경우 'winget install Graphviz.Graphviz' 또는 https://graphviz.org/download/ 에서 설치할 수 있습니다."
        )


if __name__ == "__main__":
    main()

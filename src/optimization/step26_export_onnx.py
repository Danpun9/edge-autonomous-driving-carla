"""
File: step26_export_onnx.py

Purpose:
    Export trained SMPHybridUNet checkpoints from PyTorch to ONNX.

Main Responsibilities:
    - Load ResNet34/ResNet50 U-Net checkpoints.
    - Trace the model with a 1x3x180x640 dummy input.
    - Save ONNX graphs under onnx_models/.

Notes:
    Checkpoint paths must match the encoder names configured in MODELS_TO_EXPORT.
"""

import os
import torch
from src.models.step22_smp_model import SMPHybridUNet

from src import config as project_config

# ==========================================
# 1. 경로 및 설정
# ==========================================
OUTPUT_DIR = project_config.ONNX_OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 변환할 타겟 모델 리스트
MODELS_TO_EXPORT = [
    {
        "name": "ResNet34_Aug",
        "weight_path": project_config.SMP_RESNET34_CHECKPOINT,
        "encoder": "resnet34"
    },
    {
        "name": "ResNet50_Aug",
        "weight_path": project_config.SMP_RESNET50_CHECKPOINT,
        "encoder": "resnet50"
    }
]

# 모델 입력 해상도 (추론 시와 동일하게 상단 크롭을 반영한 180x640)
INPUT_SHAPE = (1, 3, 180, 640) 

# ==========================================
# 2. ONNX Export 함수
# ==========================================
def export_to_onnx(model_info):
    device = torch.device("cpu") # ONNX 변환은 CPU에서 진행하는 것이 안정적입니다.
    
    print(f"\n🚀 [{model_info['name']}] 모델 로드 중...")
    
    # 1. 모델 아키텍처 초기화 및 가중치 로드
    model = SMPHybridUNet(encoder_name=model_info['encoder'], classes=3).to(device)
    
    if not os.path.exists(model_info['weight_path']):
        print(f"⚠️ 에러: {model_info['weight_path']} 가중치 파일이 없습니다.")
        return

    model.load_state_dict(torch.load(model_info['weight_path'], map_location=device)['model_state_dict'])
    model.eval() # 반드시 추론(Eval) 모드로 변경해야 합니다 (Dropout, BatchNorm 고정).

    # 2. 모델에 통과시킬 더미(Dummy) 입력 텐서 생성
    dummy_input = torch.randn(*INPUT_SHAPE).to(device)
    onnx_file_path = os.path.join(OUTPUT_DIR, f"{model_info['name']}.onnx")

    # 3. ONNX 포맷으로 내보내기
    print(f"🔄 ONNX 포맷으로 계산 그래프 추출 중...")
    torch.onnx.export(
        model,                              # 실행할 모델
        dummy_input,                        # 더미 입력
        onnx_file_path,                     # 저장될 파일 경로
        export_params=True,                 # 모델 가중치 포함 여부
        opset_version=11,                   # TensorRT 호환성이 가장 좋은 Opset 버전
        do_constant_folding=True,           # 상수 폴딩 최적화 적용
        input_names=['input'],              # 입력 노드 이름 지정
        output_names=['output'],            # 출력 노드 이름 지정
        dynamic_axes={                      # 배치 사이즈를 가변적(Dynamic)으로 열어둠
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✅ 변환 완료: {onnx_file_path}")

# ==========================================
# 3. 메인 파이프라인
# ==========================================
def main():
    print("==================================================")
    print("📦 PyTorch to ONNX Export 파이프라인 가동")
    print("==================================================")
    for model_info in MODELS_TO_EXPORT:
        export_to_onnx(model_info)
    print("\n🎉 모든 타겟 모델의 ONNX 추출이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    main()

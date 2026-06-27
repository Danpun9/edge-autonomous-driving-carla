"""
File: step27_quantize_onnx.py

Purpose:
    Produce FP16 and INT8 ONNX variants for the exported ResNet34 segmentation
    model.

Main Responsibilities:
    - Prepare calibration samples from the augmented U-Net dataset.
    - Convert an ONNX FP32 model to FP16.
    - Run static INT8 quantization with ONNX Runtime calibration data.

Notes:
    Writes onnx_quantized/ and _dataset_calibration/. These generated artifacts
    are excluded from Git.
"""

import os
import cv2
import numpy as np
import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import random
import shutil

from src import config as project_config

# ==========================================
# 1. 경로 및 설정
# ==========================================
MODEL_NAME = "ResNet34_Aug"
INPUT_ONNX = project_config.RESNET34_ONNX_MODEL
OUTPUT_DIR = project_config.ONNX_QUANTIZED_DIR

CALIB_IMG_DIR = os.path.join(project_config.DATASET_CALIBRATION_DIR, "images")
SOURCE_IMG_DIR = os.path.join(project_config.DATASET_AUG_UNET_DIR, "images")

CROP_Y = project_config.CROP_Y
CALIBRATION_SAMPLES = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. 캘리브레이션 데이터 준비
# ==========================================
def prepare_calibration_data():
    os.makedirs(CALIB_IMG_DIR, exist_ok=True)
    existing_files = os.listdir(CALIB_IMG_DIR)
    
    if len(existing_files) < CALIBRATION_SAMPLES:
        print(f"🔍 캘리브레이션용 이미지 {CALIBRATION_SAMPLES}장 무작위 추출 중...")
        all_imgs = [f for f in os.listdir(SOURCE_IMG_DIR) if f.endswith('.png') or f.endswith('.jpg')]
        sampled_imgs = random.sample(all_imgs, CALIBRATION_SAMPLES)
        
        for img_name in sampled_imgs:
            shutil.copy(os.path.join(SOURCE_IMG_DIR, img_name), os.path.join(CALIB_IMG_DIR, img_name))
        print("✅ 캘리브레이션 데이터 준비 완료!")

# ==========================================
# 3. ONNX Runtime 전용 Data Reader 클래스
# ==========================================
class UNetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_folder):
        self.image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        self.enum_data = iter(self.image_files)
        # ONNX 모델의 입력 노드 이름 (step26에서 'input'으로 지정함)
        self.input_name = "input" 

    def get_next(self):
        img_path = next(self.enum_data, None)
        if img_path is None:
            return None # 순회 종료
        
        # PyTorch 학습 시와 완벽하게 동일한 전처리
        image = cv2.imread(img_path)
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_cropped = frame_rgb[CROP_Y:, :, :].astype(np.float32) / 255.0
        img_tensor = np.transpose(img_cropped, (2, 0, 1)) # HWC -> CHW
        img_tensor = np.expand_dims(img_tensor, axis=0)   # 배치 차원 추가: (1, 3, 180, 640)
        
        return {self.input_name: img_tensor}

# ==========================================
# 4. 양자화 파이프라인
# ==========================================
def main():
    print("==================================================")
    print("🧠 ONNX Runtime 하드웨어 최적화 및 양자화 파이프라인")
    print("==================================================")
    
    if not os.path.exists(INPUT_ONNX):
        print(f"⚠️ 에러: {INPUT_ONNX} 파일이 존재하지 않습니다.")
        return

    # 1. 캘리브레이션 데이터 세팅
    prepare_calibration_data()
    
    # 2. FP16 (반정밀도) 변환
    fp16_model_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_FP16.onnx")
    print("\n⚙️ [1/2] FP16(반정밀도) 모델로 압축 중...")
    fp32_model = onnx.load(INPUT_ONNX)
    fp16_model = float16.convert_float_to_float16(fp32_model)
    onnx.save(fp16_model, fp16_model_path)
    print(f"✅ FP16 모델 저장 완료: {fp16_model_path}")

    # 3. INT8 (정수 정밀도) 변환
    int8_model_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_INT8.onnx")
    print("\n⚙️ [2/2] INT8(정수) 모델로 양자화 중... (캘리브레이션 데이터 100장 스캔)")
    
    data_reader = UNetCalibrationDataReader(CALIB_IMG_DIR)
    
    quantize_static(
        model_input=INPUT_ONNX,
        model_output=int8_model_path,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ, # 최신 양자화 포맷 (Quantize-Dequantize)
        op_types_to_quantize=['Conv', 'MatMul', 'Add', 'Mul'], # 주요 연산만 양자화
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8
    )
    print(f"✅ INT8 모델 저장 완료: {int8_model_path}")

if __name__ == "__main__":
    main()

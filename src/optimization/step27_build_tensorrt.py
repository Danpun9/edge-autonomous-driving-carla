"""
File: step27_build_tensorrt.py

Purpose:
    Build TensorRT FP16 and INT8 engines from the exported ResNet34 ONNX model.

Main Responsibilities:
    - Prepare calibration images for INT8 TensorRT builds.
    - Parse the ONNX graph and configure TensorRT optimization profiles.
    - Serialize FP16/INT8 engine files under tensorrt_engines/.

Notes:
    Requires NVIDIA TensorRT, PyCUDA, CUDA, and a compatible GPU runtime.
"""

import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import random
import shutil

from src import config as project_config

# ==========================================
# 1. 경로 및 하이퍼파라미터 설정
# ==========================================
ONNX_MODEL_PATH = project_config.RESNET34_ONNX_MODEL
ENGINE_DIR = project_config.TENSORRT_ENGINE_DIR
CALIB_IMG_DIR = os.path.join(project_config.DATASET_CALIBRATION_DIR, "images")
SOURCE_IMG_DIR = os.path.join(project_config.DATASET_AUG_UNET_DIR, "images")

TARGET_W, TARGET_H = 640, 360
CROP_Y = project_config.CROP_Y
BATCH_SIZE = 1
CALIBRATION_SAMPLES = 100

os.makedirs(ENGINE_DIR, exist_ok=True)

# ==========================================
# 2. 캘리브레이션 데이터셋 준비 함수
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
# 3. INT8 엔트로피 캘리브레이터 클래스 (Entropy Calibrator)
# ==========================================
class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file="resnet34_calibration.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.image_files = [os.path.join(CALIB_IMG_DIR, f) for f in os.listdir(CALIB_IMG_DIR)]
        self.batch_size = BATCH_SIZE
        self.current_index = 0
        
        # 모델 입력 텐서(1, 3, 180, 640) 크기에 맞춰 GPU 메모리 할당
        self.device_input = cuda.mem_alloc(self.batch_size * 3 * 180 * 640 * 4) # float32(4 bytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.image_files):
            return None # 캘리브레이션 종료

        batch_imgs = []
        for i in range(self.batch_size):
            img_path = self.image_files[self.current_index + i]
            image = cv2.imread(img_path)
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # PyTorch 학습 시와 완벽하게 동일한 전처리 적용 (크롭 및 정규화)
            img_cropped = frame_rgb[CROP_Y:, :, :].astype(np.float32) / 255.0
            img_tensor = np.transpose(img_cropped, (2, 0, 1)) # HWC to CHW
            batch_imgs.append(img_tensor)

        batch_data = np.ascontiguousarray(np.stack(batch_imgs))
        cuda.memcpy_htod(self.device_input, batch_data) # CPU 데이터를 GPU로 복사
        self.current_index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# ==========================================
# 4. TensorRT 엔진 빌더 함수
# ==========================================
def build_engine(onnx_path, engine_path, precision='FP16'):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    # 동적 배치 및 네트워크 정의 설정
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    
    # 워크스페이스 메모리 설정 (4GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))

    # ONNX 파싱 (이전 단계에서 수정한 코드)
    if not parser.parse_from_file(onnx_path):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return None

    # ──────────────────────────────────────────────────────────
    # ⭐ [추가] Dynamic Shape 대응을 위한 최적화 프로필 설정
    # ──────────────────────────────────────────────────────────
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0) # 모델의 첫 번째 입력 텐서 선택
    
    # 스크립트 상단 설정 기준 크기: (Batch=1, Channels=3, Height=180, Width=640)
    # 최소(min), 최적(opt), 최대(max) 크기를 모두 동일하게 고정하여 배치 1 크기로 최적화합니다.
    target_shape = (BATCH_SIZE, 3, 180, 640)
    profile.set_shape(input_tensor.name, min=target_shape, opt=target_shape, max=target_shape)
    config.add_optimization_profile(profile)
    # ──────────────────────────────────────────────────────────

    print(f"\n⚙️ [{precision}] 정밀도로 엔진 빌드 중... (몇 분 정도 소요될 수 있습니다)")
    
    # FP16 설정
    if precision == 'FP16' and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        
    # INT8 설정 및 캘리브레이터 부착
    elif precision == 'INT8' and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Int8Calibrator(cache_file="resnet34_int8.cache")
        print("📊 INT8 캘리브레이션 활성화 (데이터셋 스캔 중...)")

    # 엔진 빌드 및 직렬화
    serialized_engine = builder.build_serialized_network(network, config)
    
    # 빌드 실패 시 예외 처리 추가 (None 에러 방지)
    if serialized_engine is None:
        print(f"❌ [{precision}] 엔진 빌드에 실패했습니다. 코드가 올바르게 입력되었는지 확인해 주세요.")
        return None

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"✅ {precision} 엔진 저장 완료: {engine_path}")

def main():
    print("==================================================")
    print("🛠️ TensorRT 양자화 컴파일러 가동")
    print("==================================================")
    
    prepare_calibration_data()
    
    # 1. FP16 엔진 빌드
    build_engine(ONNX_MODEL_PATH, os.path.join(ENGINE_DIR, "ResNet34_Aug_FP16.engine"), precision='FP16')
    
    # 2. INT8 엔진 빌드 (캘리브레이션 포함)
    build_engine(ONNX_MODEL_PATH, os.path.join(ENGINE_DIR, "ResNet34_Aug_INT8.engine"), precision='INT8')

if __name__ == "__main__":
    main()

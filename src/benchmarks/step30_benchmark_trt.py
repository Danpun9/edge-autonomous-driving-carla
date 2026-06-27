"""
File: step30_benchmark_trt.py

Purpose:
    Benchmark native TensorRT segmentation engines on the AI-Hub evaluation set.

Main Responsibilities:
    - Load TensorRT FP16 and INT8 engine files.
    - Run CUDA memory transfers and TensorRT inference.
    - Report mIoU, FPS, VRAM, and power metrics.

Notes:
    Requires TensorRT 10.x-style APIs, PyCUDA, NVML, and generated engine files.
"""

import os
import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pynvml
from tqdm import tqdm

from src import config as project_config

# ==========================================
# 1. 경로 및 설정
# ==========================================
DATASET_DIR = project_config.DATASET_AIHUB_EVAL_DIR
IMG_DIR = os.path.join(DATASET_DIR, "images")
MSK_DIR = os.path.join(DATASET_DIR, "masks")
OUTPUT_DIR = project_config.BENCHMARK_OUTPUT_DIR
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

MODELS_TO_EVALUATE = {
    "ResNet34 (FP16 Native TRT)": project_config.TENSORRT_FP16_ENGINE,
    "ResNet34 (INT8 Native TRT)": project_config.TENSORRT_INT8_ENGINE,
}

TARGET_W, TARGET_H = 640, 360
CROP_Y = project_config.CROP_Y
DILATION_KERNEL = np.ones((15, 15), np.uint8)
NUM_VISUALIZE = 5
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ==========================================
# 2. 메인 벤치마크 파이프라인
# ==========================================
def main():
    print("🚀 WSL2 Native TensorRT Benchmark 파이프라인 가동 (TRT 10.x API)")
    
    try:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
        print(f"🖥️ 감지된 GPU: {gpu_name}")
    except Exception as e:
        print(f"⚠️ NVML 초기화 실패: {e}")
        return

    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])
    if not img_files: return

    sample_indices = np.linspace(0, len(img_files) - 1, NUM_VISUALIZE, dtype=int)
    sample_files = [img_files[i] for i in sample_indices]
    metrics_report = {}

    for model_name, engine_path in MODELS_TO_EVALUATE.items():
        if not os.path.exists(engine_path):
            print(f"\n⚠️ {engine_path} 파일이 없습니다. step27 빌드를 확인하세요.")
            continue
            
        print(f"\n📊 [{model_name}] 엔진 로드 및 GPU 메모리 할당 중...")
        
        # 엔진 역직렬화 및 컨텍스트 생성
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        # ⭐ [변경됨] TRT 10.x 텐서 이름 및 Shape 설정
        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)
        
        input_shape = (1, 3, 180, 640)
        output_shape = (1, 3, 180, 640) 
        
        # Dynamic Shape 프로필을 위해 입력 크기 명시
        context.set_input_shape(input_name, input_shape)

        # 메모리 버퍼 생성 (Page-locked 메모리)
        h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
        h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
        
        # GPU(Device) 메모리 할당
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        
        # ⭐ [변경됨] TRT 10.x 방식으로 메모리 주소 매핑
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))
        
        stream = cuda.Stream()

        total_inter, total_union, total_time = 0, 0, 0.0
        vram_usage_list, power_usage_list = [], []
        safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")

        for img_name in tqdm(img_files, desc=f"Evaluating {model_name}"):
            img_path = os.path.join(IMG_DIR, img_name)
            msk_path = os.path.join(MSK_DIR, img_name.replace('.jpg', '.png'))

            # 이미지 전처리
            image = cv2.imread(img_path)
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_cropped = frame_rgb[CROP_Y:, :, :].astype(np.float32) / 255.0
            img_tensor = np.transpose(img_cropped, (2, 0, 1)).ravel()

            # CPU -> Page-locked 복사
            np.copyto(h_input, img_tensor)

            gt_mask_full = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            gt_cropped = gt_mask_full[CROP_Y:, :]
            gt_dilated = cv2.dilate(gt_cropped, DILATION_KERNEL, iterations=1)
            gt_inds = (gt_dilated == 1)

            # --- [순수 GPU 추론 시간 측정] ---
            start_time = time.time()
            
            # CPU -> GPU 비동기 복사
            cuda.memcpy_htod_async(d_input, h_input, stream)
            
            # ⭐ [변경됨] execute_async_v3 호출 (bindings 인자 제거)
            context.execute_async_v3(stream_handle=stream.handle)
            
            # GPU -> CPU 비동기 복사
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
            
            total_time += (time.time() - start_time)
            # --- [추론 완료] ---

            # 하드웨어 상태 폴링
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(gpu_handle)
            vram_usage_list.append(mem_info.used / (1024 ** 2))
            power_usage_list.append(power_mw / 1000.0)

            # 결과 후처리
            output_tensor = h_output.reshape(output_shape)
            pred_mask_cropped = np.argmax(output_tensor[0], axis=0).squeeze()
            pred_inds = (pred_mask_cropped == 1)

            total_inter += np.logical_and(pred_inds, gt_inds).sum()
            total_union += np.logical_or(pred_inds, gt_inds).sum()

            # 샘플 개별 저장
            if img_name in sample_files:
                overlay = image.copy()
                pred_mask_full = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
                pred_mask_full[CROP_Y:, :] = pred_mask_cropped
                overlay[pred_mask_full == 1] = [0, 0, 255]
                cv2.imwrite(os.path.join(SAMPLE_DIR, f"{img_name}_{safe_model_name}.jpg"), overlay)

        mIoU = (total_inter / total_union) * 100 if total_union > 0 else 0
        fps = len(img_files) / total_time if total_time > 0 else 0
        avg_vram = sum(vram_usage_list) / len(vram_usage_list)
        avg_power = sum(power_usage_list) / len(power_usage_list)

        metrics_report[model_name] = {
            "mIoU": mIoU, "FPS": fps, "VRAM_MB": avg_vram, "Power_W": avg_power
        }

    pynvml.nvmlShutdown()

    print("\n" + "="*85)
    print("📊 [Native TensorRT Benchmark Report]")
    print("="*85)
    print(f"{'Model Name':<30} | {'mIoU (%)':<10} | {'FPS':<10} | {'VRAM (MB)':<12} | {'Power (W)':<10}")
    print("-" * 85)
    for name, metrics in metrics_report.items():
        print(f"{name:<30} | {metrics['mIoU']:<10.2f} | {metrics['FPS']:<10.1f} | {metrics['VRAM_MB']:<12.1f} | {metrics['Power_W']:<10.1f}")
    print("="*85)
    print("✅ 평가가 완료되었습니다. step29 시각화 스크립트를 실행해 결과를 병합하세요.")

if __name__ == "__main__":
    main()

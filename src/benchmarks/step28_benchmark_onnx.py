"""
File: step28_benchmark_onnx.py

Purpose:
    Benchmark ONNX Runtime variants of the ResNet34 segmentation model.

Main Responsibilities:
    - Load FP32, FP16, and INT8 ONNX models.
    - Measure mIoU, FPS, VRAM, and power on _dataset_aihub_eval/.
    - Save sample overlays under benchmark_results/samples/.

Notes:
    Requires ONNX Runtime GPU and NVML access for hardware metrics.
"""

import os
import cv2
import time
import numpy as np
import onnxruntime as ort
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
    "ResNet34 (FP32 Original)": project_config.RESNET34_ONNX_MODEL,
    "ResNet34 (FP16 Half)": os.path.join(project_config.ONNX_QUANTIZED_DIR, "ResNet34_Aug_FP16.onnx"),
    "ResNet34 (INT8 Quantized)": os.path.join(project_config.ONNX_QUANTIZED_DIR, "ResNet34_Aug_INT8.onnx"),
}

TARGET_W, TARGET_H = 640, 360
CROP_Y = project_config.CROP_Y
DILATION_KERNEL = np.ones((15, 15), np.uint8)
NUM_VISUALIZE = 5

PROVIDERS = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider'
]

# ==========================================
# 2. 메인 벤치마크 파이프라인
# ==========================================
def main():
    print("🚀 ONNX Runtime Hardware Benchmark 파이프라인 가동 (평가 전용)")
    
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
    saved_gt_inputs = set()
    metrics_report = {}

    for model_name, onnx_path in MODELS_TO_EVALUATE.items():
        if not os.path.exists(onnx_path): continue
        print(f"\n📊 [{model_name}] 모델 로드 중...")
        
        session = ort.InferenceSession(onnx_path, providers=PROVIDERS)
        input_name = session.get_inputs()[0].name

        total_inter, total_union, total_time = 0, 0, 0.0
        vram_usage_list, power_usage_list = [], []
        
        safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")

        for img_name in tqdm(img_files, desc=f"Evaluating {model_name}"):
            img_path = os.path.join(IMG_DIR, img_name)
            msk_path = os.path.join(MSK_DIR, img_name.replace('.jpg', '.png'))

            image = cv2.imread(img_path)
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_cropped = frame_rgb[CROP_Y:, :, :].astype(np.float32) / 255.0
            img_tensor = np.transpose(img_cropped, (2, 0, 1))
            img_tensor = np.expand_dims(img_tensor, axis=0)

            # 동적 타입 캐스팅 (에러 방지)
            input_type = session.get_inputs()[0].type
            if 'float16' in input_type:
                img_tensor = img_tensor.astype(np.float16)
            else:
                img_tensor = img_tensor.astype(np.float32)

            gt_mask_full = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            gt_cropped = gt_mask_full[CROP_Y:, :]
            gt_dilated = cv2.dilate(gt_cropped, DILATION_KERNEL, iterations=1)
            gt_inds = (gt_dilated == 1)

            start_time = time.time()
            outputs = session.run(None, {input_name: img_tensor})
            total_time += (time.time() - start_time)

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(gpu_handle)
            vram_usage_list.append(mem_info.used / (1024 ** 2))
            power_usage_list.append(power_mw / 1000.0)

            pred_mask_cropped = np.argmax(outputs[0], axis=1).squeeze()
            pred_inds = (pred_mask_cropped == 1)

            total_inter += np.logical_and(pred_inds, gt_inds).sum()
            total_union += np.logical_or(pred_inds, gt_inds).sum()

            # [변경됨] 시각화 이미지를 파일로 즉시 저장
            if img_name in sample_files:
                overlay = image.copy()
                pred_mask_full = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
                pred_mask_full[CROP_Y:, :] = pred_mask_cropped
                overlay[pred_mask_full == 1] = [0, 0, 255]
                cv2.imwrite(os.path.join(SAMPLE_DIR, f"{img_name}_{safe_model_name}.jpg"), overlay)
                
                # Input과 GT는 한 번만 저장
                if img_name not in saved_gt_inputs:
                    cv2.imwrite(os.path.join(SAMPLE_DIR, f"{img_name}_Input.jpg"), image)
                    gt_overlay = image.copy()
                    gt_overlay[gt_mask_full == 1] = [0, 255, 0]
                    cv2.imwrite(os.path.join(SAMPLE_DIR, f"{img_name}_GT.jpg"), gt_overlay)
                    saved_gt_inputs.add(img_name)

        mIoU = (total_inter / total_union) * 100 if total_union > 0 else 0
        fps = len(img_files) / total_time if total_time > 0 else 0
        avg_vram = sum(vram_usage_list) / len(vram_usage_list)
        avg_power = sum(power_usage_list) / len(power_usage_list)

        metrics_report[model_name] = {
            "mIoU": mIoU, "FPS": fps, "VRAM_MB": avg_vram, "Power_W": avg_power
        }

    pynvml.nvmlShutdown()

    print("\n" + "="*85)
    print("📊 [ONNX Quantization Benchmark Report]")
    print("="*85)
    print(f"{'Model Name':<30} | {'mIoU (%)':<10} | {'FPS':<10} | {'VRAM (MB)':<12} | {'Power (W)':<10}")
    print("-" * 85)
    for name, metrics in metrics_report.items():
        print(f"{name:<30} | {metrics['mIoU']:<10.2f} | {metrics['FPS']:<10.1f} | {metrics['VRAM_MB']:<12.1f} | {metrics['Power_W']:<10.1f}")
    print("="*85)
    print("✅ 샘플 이미지가 개별 저장되었습니다. 시각화 렌더링 스크립트를 실행해 주세요.")

if __name__ == "__main__":
    main()

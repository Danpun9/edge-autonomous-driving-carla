"""
File: step31_grand_benchmark.py

Purpose:
    Run an integrated ONNX/TensorRT benchmark for edge inference comparison.

Main Responsibilities:
    - Evaluate ONNX FP32/FP16/INT8 and TensorRT FP16/INT8 variants.
    - Measure mIoU, FPS, VRAM, and power usage.
    - Save a multi-model visual comparison under benchmark_results/.

Notes:
    Requires GPU runtimes for both ONNX Runtime and TensorRT. This is the main
    benchmark script reflected in the final presentation.
"""

import os
import cv2
import time
import gc
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnxruntime as ort
import pynvml
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

from src import config as project_config

# 백그라운드 렌더링 모드 (메모리 누수 방지)
matplotlib.use('Agg')

# ==========================================
# 1. 환경 설정 및 평가 라인업 구축
# ==========================================
DATASET_DIR = project_config.DATASET_AIHUB_EVAL_DIR
IMG_DIR = os.path.join(DATASET_DIR, "images")
MSK_DIR = os.path.join(DATASET_DIR, "masks")
OUTPUT_DIR = project_config.BENCHMARK_OUTPUT_DIR
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "grand_samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# 5-Tier 그랜드 평가 라인업 (ONNX 3종 + TRT 2종)
MODELS_TO_EVALUATE = {
    "ResNet34_ONNX_FP32": project_config.RESNET34_ONNX_MODEL,
    "ResNet34_ONNX_FP16": os.path.join(project_config.ONNX_QUANTIZED_DIR, "ResNet34_Aug_FP16.onnx"),
    "ResNet34_ONNX_INT8": os.path.join(project_config.ONNX_QUANTIZED_DIR, "ResNet34_Aug_INT8.onnx"),
    "ResNet34_TRT_FP16": project_config.TENSORRT_FP16_ENGINE,
    "ResNet34_TRT_INT8": project_config.TENSORRT_INT8_ENGINE,
}

TARGET_W, TARGET_H = 640, 360
CROP_Y = project_config.CROP_Y
DILATION_KERNEL = np.ones((15, 15), np.uint8)
NUM_VISUALIZE = 5

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ORT_PROVIDERS = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    })
]

# ==========================================
# 2. 통합 벤치마크 엔진
# ==========================================
def main():
    print("🚀 [Step 31] 5-Tier Grand Benchmark 파이프라인 가동")
    
    try:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
        print(f"🖥️ 테스트 환경: {gpu_name} (WSL2)")
    except Exception as e:
        print(f"⚠️ NVML 초기화 실패: {e}")
        return

    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])
    if not img_files: return

    sample_indices = np.linspace(0, len(img_files) - 1, NUM_VISUALIZE, dtype=int)
    sample_files = [img_files[i] for i in sample_indices]
    
    metrics_report = {}
    saved_gt_inputs = set()

    for model_name, model_path in MODELS_TO_EVALUATE.items():
        if not os.path.exists(model_path):
            print(f"\n⚠️ {model_path} 파일 누락. 평가를 건너뜁니다.")
            continue
            
        print(f"\n📊 [{model_name}] 평가 준비 중...")
        is_onnx = model_path.endswith('.onnx')
        
        # --- [1. 프레임워크별 초기화] ---
        if is_onnx:
            session = ort.InferenceSession(model_path, providers=ORT_PROVIDERS)
            input_name = session.get_inputs()[0].name
            input_type = session.get_inputs()[0].type
        else:
            with open(model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()
            
            # TRT 10.x API
            trt_in_name = engine.get_tensor_name(0)
            trt_out_name = engine.get_tensor_name(1)
            input_shape = (1, 3, 180, 640)
            output_shape = (1, 3, 180, 640) 
            context.set_input_shape(trt_in_name, input_shape)

            h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
            h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
            d_input = cuda.mem_alloc(h_input.nbytes)
            d_output = cuda.mem_alloc(h_output.nbytes)
            
            context.set_tensor_address(trt_in_name, int(d_input))
            context.set_tensor_address(trt_out_name, int(d_output))
            stream = cuda.Stream()

        total_inter, total_union, total_time = 0, 0, 0.0
        vram_usage_list, power_usage_list = [], []

        # --- [2. 추론 루프] ---
        for img_name in tqdm(img_files, desc=f"Evaluating {model_name}"):
            img_path = os.path.join(IMG_DIR, img_name)
            msk_path = os.path.join(MSK_DIR, img_name.replace('.jpg', '.png'))

            # 이미지 전처리
            image = cv2.imread(img_path)
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_cropped = frame_rgb[CROP_Y:, :, :].astype(np.float32) / 255.0
            
            # 정답지 로드 및 팽창
            gt_mask_full = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            gt_cropped = gt_mask_full[CROP_Y:, :]
            gt_dilated = cv2.dilate(gt_cropped, DILATION_KERNEL, iterations=1)
            gt_inds = (gt_dilated == 1)

            start_time = time.time()
            
            if is_onnx:
                img_tensor = np.transpose(img_cropped, (2, 0, 1))
                img_tensor = np.expand_dims(img_tensor, axis=0)
                if 'float16' in input_type:
                    img_tensor = img_tensor.astype(np.float16)
                
                outputs = session.run(None, {input_name: img_tensor})
                pred_mask_cropped = np.argmax(outputs[0], axis=1).squeeze()
            else:
                img_tensor = np.transpose(img_cropped, (2, 0, 1)).ravel()
                np.copyto(h_input, img_tensor)
                cuda.memcpy_htod_async(d_input, h_input, stream)
                context.execute_async_v3(stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()
                
                output_tensor = h_output.reshape(output_shape)
                pred_mask_cropped = np.argmax(output_tensor[0], axis=0).squeeze()

            total_time += (time.time() - start_time)

            # 리소스 추적
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(gpu_handle)
            vram_usage_list.append(mem_info.used / (1024 ** 2))
            power_usage_list.append(power_mw / 1000.0)

            # 정확도 집계
            pred_inds = (pred_mask_cropped == 1)
            total_inter += np.logical_and(pred_inds, gt_inds).sum()
            total_union += np.logical_or(pred_inds, gt_inds).sum()

            # 시각화 렌더링용 개별 저장
            if img_name in sample_files:
                overlay = image.copy()
                pred_mask_full = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
                pred_mask_full[CROP_Y:, :] = pred_mask_cropped
                overlay[pred_mask_full == 1] = [0, 0, 255]
                cv2.imwrite(os.path.join(SAMPLE_DIR, f"{img_name}_{model_name}.jpg"), overlay)
                
                if img_name not in saved_gt_inputs:
                    cv2.imwrite(os.path.join(SAMPLE_DIR, f"{img_name}_Input.jpg"), image)
                    gt_overlay = image.copy()
                    gt_overlay[gt_mask_full == 1] = [0, 255, 0]
                    cv2.imwrite(os.path.join(SAMPLE_DIR, f"{img_name}_GT.jpg"), gt_overlay)
                    saved_gt_inputs.add(img_name)

        # 결과 저장
        metrics_report[model_name] = {
            "mIoU": (total_inter / total_union) * 100 if total_union > 0 else 0,
            "FPS": len(img_files) / total_time if total_time > 0 else 0,
            "VRAM_MB": sum(vram_usage_list) / len(vram_usage_list),
            "Power_W": sum(power_usage_list) / len(power_usage_list)
        }

        # --- [3. VRAM 강제 초기화 (메모리 누수 방지)] ---
        if is_onnx:
            del session
        else:
            d_input.free()
            d_output.free()
            del context, engine, runtime
        
        gc.collect()
        time.sleep(1.5) # GPU 캐시가 비워질 여유 시간 제공

    pynvml.nvmlShutdown()

    # ==========================================
    # 3. 콘솔 결과 및 7-Col 파노라마 시각화 렌더링
    # ==========================================
    print("\n" + "="*85)
    print("🏆 [Grand AI Benchmark Report] - Sim2Real Edge Optimization")
    print("="*85)
    print(f"{'Model Name':<25} | {'mIoU (%)':<10} | {'FPS':<10} | {'VRAM (MB)':<12} | {'Power (W)':<10}")
    print("-" * 85)
    for name, metrics in metrics_report.items():
        print(f"{name:<25} | {metrics['mIoU']:<10.2f} | {metrics['FPS']:<10.1f} | {metrics['VRAM_MB']:<12.1f} | {metrics['Power_W']:<10.1f}")
    print("="*85)

    print("\n🎨 그랜드 파노라마 시각화 렌더링 중...")
    model_keys = list(MODELS_TO_EVALUATE.keys())
    col_titles = ["Input", "Ground Truth"] + [k.replace("ResNet34_", "") for k in model_keys]
    num_cols = len(col_titles)
    
    fig, axes = plt.subplots(nrows=NUM_VISUALIZE, ncols=num_cols, figsize=(4 * num_cols, 12))
    
    for row, img_name in enumerate(sample_files):
        # 1. Input과 GT 로드
        images_to_plot = [
            cv2.cvtColor(cv2.imread(os.path.join(SAMPLE_DIR, f"{img_name}_Input.jpg")), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(cv2.imread(os.path.join(SAMPLE_DIR, f"{img_name}_GT.jpg")), cv2.COLOR_BGR2RGB)
        ]
        # 2. 5개의 모델 결과 로드
        for key in model_keys:
            img_path = os.path.join(SAMPLE_DIR, f"{img_name}_{key}.jpg")
            img = cv2.imread(img_path) if os.path.exists(img_path) else np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
            images_to_plot.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        for col, img_plot in enumerate(images_to_plot):
            axes[row, col].imshow(img_plot)
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=14, fontweight='bold')

    plt.tight_layout(pad=1.0)
    vis_path = os.path.join(OUTPUT_DIR, "grand_benchmark_visualization.png")
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 그랜드 시각화 저장 완료: {vis_path}")

if __name__ == "__main__":
    main()

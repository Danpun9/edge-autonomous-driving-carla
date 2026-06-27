"""
File: step32_make_inference_video.py

Purpose:
    Render an inference video from an image sequence using TensorRT lane
    detection and YOLO object detection.

Main Responsibilities:
    - Load a TensorRT INT8 lane-segmentation engine.
    - Run YOLO on each frame for object overlays.
    - Save the rendered video under benchmark_results/.

Notes:
    Requires TensorRT/PyCUDA and the configured engine/model paths.
"""

import os
import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from ultralytics import YOLO
from tqdm import tqdm

from src import config as project_config

# ==========================================
# 1. 경로 및 설정
# ==========================================
DATASET_DIR = os.path.join(project_config.DATASET_OTHER_SIM_DIR, "images")
OUTPUT_VIDEO_PATH = os.path.join(project_config.BENCHMARK_OUTPUT_DIR, "sim_inference_video.mp4")
TRT_ENGINE_PATH = project_config.TENSORRT_INT8_ENGINE
YOLO_MODEL_PATH = project_config.YOLO_MODEL_PATH

TARGET_W, TARGET_H = 640, 360
CROP_Y = project_config.CROP_Y
FPS_OUT = 30  # 출력 비디오의 FPS (보통 30으로 설정)

# ==========================================
# 2. 텐서 코어 전용 차선 인식 클래스 (TRT 10.x)
# ==========================================
class TRTLaneDetector:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = (1, 3, 180, 640)
        self.output_shape = (1, 3, 180, 640)
        self.context.set_input_shape(self.input_name, self.input_shape)

        self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        self.stream = cuda.Stream()

    def predict(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_cropped = frame_rgb[CROP_Y:, :, :].astype(np.float32) / 255.0
        img_tensor = np.transpose(img_cropped, (2, 0, 1)).ravel()

        np.copyto(self.h_input, img_tensor)
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        output_tensor = self.h_output.reshape(self.output_shape)
        pred_mask_cropped = np.argmax(output_tensor[0], axis=0).squeeze().astype(np.uint8)
        
        full_mask = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
        full_mask[CROP_Y:, :] = pred_mask_cropped
        return full_mask

    def cleanup(self):
        self.d_input.free()
        self.d_output.free()

# ==========================================
# 3. 메인 영상화 파이프라인
# ==========================================
def main():
    print("🎬 AI 통합 추론 및 영상 렌더링 파이프라인 가동")
    
    if not os.path.exists(DATASET_DIR):
        print(f"⚠️ {DATASET_DIR} 폴더를 찾을 수 없습니다.")
        return

    # 1. 이미지 로드 및 정렬 (시간순)
    # 이미지 이름이 숫자 형태(예: 0001.jpg)인 경우를 대비한 안전한 정렬
    img_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(('.jpg', '.png'))]
    img_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)) 
    
    if not img_files:
        print(f"⚠️ {DATASET_DIR} 폴더 내에 이미지(jpg/png)가 없습니다.")
        return
    
    print(f"총 {len(img_files)}장의 이미지를 처리합니다.")

    # 2. 비디오 라이터(VideoWriter) 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 범용적인 MP4 코덱
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS_OUT, (TARGET_W, TARGET_H))

    # 3. 모델 로드
    lane_detector = TRTLaneDetector(TRT_ENGINE_PATH)
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    total_time = 0.0

    # 4. 순차적 추론 및 영상 압축
    for img_name in tqdm(img_files, desc="Rendering Video"):
        img_path = os.path.join(DATASET_DIR, img_name)
        frame = cv2.imread(img_path)
        
        # 해상도 일치화 방어 코드 (만약 640x360이 아니라면 강제 리사이즈)
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H))

        start_time = time.time()
        
        # --- [추론 파이프라인] ---
        # 1. 차선 인식 (TRT INT8)
        lane_mask = lane_detector.predict(frame)
        
        # 2. 객체 인식 (YOLO) - 차량, 사람, 신호등 등 타겟팅
        # classes: 0(person), 2(car), 3(motorcycle), 5(bus), 7(truck), 9(traffic light)
        yolo_results = yolo_model(frame, verbose=False, classes=[0, 2, 3, 5, 7, 9]) 
        
        total_time += (time.time() - start_time)

        # --- [오버레이 시각화] ---
        # 1. 차선 붉은색 오버레이
        overlay = frame.copy()
        overlay[lane_mask == 1] = [0, 0, 255]
        
        # 약간의 투명도(Alpha Blending)를 주어 자연스럽게 합성
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # 2. YOLO 바운딩 박스 그리기
        frame = yolo_results[0].plot(img=frame)
        
        # 프레임을 비디오에 기록
        out.write(frame)

    # 5. 리소스 정리
    out.release()
    lane_detector.cleanup()
    
    avg_fps = len(img_files) / total_time
    print(f"\n✅ 영상 렌더링이 성공적으로 완료되었습니다!")
    print(f"저장 경로: {OUTPUT_VIDEO_PATH}")
    print(f"추론 평균 FPS (순수 GPU 타임): {avg_fps:.1f}")

if __name__ == "__main__":
    main()

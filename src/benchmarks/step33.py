"""
File: step33.py

Purpose:
    Render a combined AI driving-perception video from a video input file.

Main Responsibilities:
    - Load a TensorRT INT8 lane detector and a YOLO model.
    - Resize frames to the project inference resolution.
    - Overlay lane masks and YOLO detections into an output MP4.

Notes:
    The generic file name is historical. It expects _dataset_other_sim/drive_test2.mp4
    and writes benchmark_results/sim_driving_result.mp4.
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
# 1. 텐서 코어 전용 차선 인식 클래스 (재사용)
# ==========================================
class TRTLaneDetector:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        cropped_height = project_config.IMAGE_HEIGHT - project_config.CROP_Y
        self.input_shape = (1, 3, cropped_height, project_config.IMAGE_WIDTH)
        self.output_shape = (1, 3, cropped_height, project_config.IMAGE_WIDTH)
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
        img_cropped = frame_rgb[project_config.CROP_Y:, :, :].astype(np.float32) / 255.0
        img_tensor = np.transpose(img_cropped, (2, 0, 1)).ravel()

        np.copyto(self.h_input, img_tensor)
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        output_tensor = self.h_output.reshape(self.output_shape)
        pred_mask_cropped = np.argmax(output_tensor[0], axis=0).squeeze().astype(np.uint8)
        
        full_mask = np.zeros((project_config.IMAGE_HEIGHT, project_config.IMAGE_WIDTH), dtype=np.uint8)
        full_mask[project_config.CROP_Y:, :] = pred_mask_cropped
        return full_mask

    def cleanup(self):
        self.d_input.free()
        self.d_output.free()

# ==========================================
# 2. 비디오 처리 메인 파이프라인
# ==========================================
def main():
    print("🎬 비디오 기반 통합 AI 추론 파이프라인 가동")

    # 1. 경로 설정
    VIDEO_INPUT_PATH = project_config.OTHER_SIM_DRIVE_TEST_VIDEO
    OUTPUT_DIR = project_config.BENCHMARK_OUTPUT_DIR
    VIDEO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sim_driving_result.mp4")
    
    if not os.path.exists(VIDEO_INPUT_PATH):
        print(f"⚠️ 에러: {VIDEO_INPUT_PATH} 파일을 찾을 수 없습니다.")
        return

    # 2. 모델 로드
    print("🧠 AI 모델 적재 중...")
    lane_detector = TRTLaneDetector(project_config.TENSORRT_INT8_ENGINE)
    yolo_model = YOLO(project_config.YOLO_MODEL_PATH)

    # 3. 비디오 객체 초기화
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 원본 해상도와 관계없이 우리 모델 규격(640x360)으로 강제 리사이즈 및 저장
    TARGET_W, TARGET_H = project_config.IMAGE_WIDTH, project_config.IMAGE_HEIGHT
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4 코덱 설정
    out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (TARGET_W, TARGET_H))

    print(f"📼 비디오 처리 시작: 총 {total_frames} 프레임 (타겟 해상도: {TARGET_W}x{TARGET_H})")
    
    processing_times = []

    # 4. 프레임 단위 추론 루프
    for _ in tqdm(range(total_frames), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret:
            break

        # 모델 규격에 맞게 리사이즈
        frame_resized = cv2.resize(frame, (TARGET_W, TARGET_H))
        
        start_time = time.time()

        # --- [AI 추론] ---
        # 1. 차선 인식
        lane_mask = lane_detector.predict(frame_resized)
        
        # 2. 객체 인식 (보행자, 차량 등 주요 클래스 필터링)
        yolo_results = yolo_model(frame_resized, verbose=False, classes=[0, 1, 2, 3, 5, 7]) 
        
        processing_times.append(time.time() - start_time)

        # --- [시각화 오버레이] ---
        # 차선 그리기 (빨간색 반투명 오버레이)
        color_mask = np.zeros_like(frame_resized)
        color_mask[lane_mask == 1] = [0, 0, 255]
        annotated_frame = cv2.addWeighted(frame_resized, 1.0, color_mask, 0.7, 0)
        
        # YOLO 바운딩 박스 그리기
        annotated_frame = yolo_results[0].plot(img=annotated_frame)

        # 결과 영상에 쓰기
        out.write(annotated_frame)

    # 5. 리소스 해제 및 결과 출력
    cap.release()
    out.release()
    lane_detector.cleanup()

    avg_fps = 1.0 / (sum(processing_times) / len(processing_times)) if processing_times else 0

    print("\n" + "="*50)
    print("✅ 비디오 렌더링 완료!")
    print(f"저장 경로: {VIDEO_OUTPUT_PATH}")
    print(f"순수 AI 추론 평균 FPS: {avg_fps:.1f}")
    print("="*50)

if __name__ == "__main__":
    main()

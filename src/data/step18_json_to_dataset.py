"""
File: step18_json_to_dataset.py

Purpose:
    Convert manually labeled real-road JSON files into U-Net and YOLO datasets.

Main Responsibilities:
    - Resize real images to the project target resolution.
    - Convert polygon/box annotations into YOLO labels and class-index masks.
    - Save images, labels, masks, and debug overlays under _dataset_real_processing/.

Notes:
    Input JSON files are expected to follow the AnyLabeling-style shape schema.
"""

import os
import json
import cv2
import numpy as np

from src import config as project_config

# ==========================================
# 1. 경로 및 하이퍼파라미터 설정
# ==========================================
BASE_DIR = project_config.DATASET_REAL_DIR
INPUT_FOLDERS = ["case1_images", "case2_images", "case3_images", "case4_images"]

OUTPUT_BASE = project_config.DATASET_REAL_PROCESSED_DIR
OUT_IMG_DIR = os.path.join(OUTPUT_BASE, "images")
OUT_LBL_DIR = os.path.join(OUTPUT_BASE, "labels") # YOLO용 txt
OUT_MSK_DIR = os.path.join(OUTPUT_BASE, "masks")  # U-Net용 png (0, 1, 2)
OUT_DBG_DIR = os.path.join(OUTPUT_BASE, "debug")  # 디버깅용 시각화

# 출력 폴더 생성
for folder in [OUT_IMG_DIR, OUT_LBL_DIR, OUT_MSK_DIR, OUT_DBG_DIR]:
    os.makedirs(folder, exist_ok=True)

# 원본 및 타겟 해상도 설정
ORIGINAL_W, ORIGINAL_H = 1920, 1080
TARGET_W, TARGET_H = 640, 360
SCALE_X = TARGET_W / ORIGINAL_W
SCALE_Y = TARGET_H / ORIGINAL_H

# 라벨 매핑 딕셔너리
YOLO_CLASSES = {'vehicle': 0, 'walker': 1, 'traffic_light': 2}
UNET_CLASSES = {'lane': 1, 'crosswalk': 2}
LANE_THICKNESS = 10 # 640x360 기준 차선 두께 (원본 1920 기준 약 30px 수준)

def main():
    print("🚀 AnyLabeling JSON -> Dual Dataset 변환 엔진 가동 시작...")
    total_processed = 0

    for case_folder in INPUT_FOLDERS:
        folder_path = os.path.join(BASE_DIR, case_folder)
        if not os.path.exists(folder_path):
            print(f"⚠️ 폴더를 찾을 수 없습니다: {folder_path} (건너뜁니다)")
            continue
        
        # 해당 폴더 내의 json 파일 목록 추출
        json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])
        
        for json_file in json_files:
            base_name = os.path.splitext(json_file)[0] # 예: "real_0001"
            json_path = os.path.join(folder_path, json_file)
            img_path = os.path.join(folder_path, f"{base_name}.jpg")
            
            if not os.path.exists(img_path):
                continue # 짝이 맞는 이미지가 없으면 패스

            # 중복 방지를 위한 새로운 고유 파일명 생성 (예: case1_real_0001)
            # 폴더명 "case1_images" 에서 "case1"만 추출
            case_prefix = case_folder.split('_')[0] 
            new_file_name = f"{case_prefix}_{base_name}"

            # ---------------------------------------------------------
            # [단계 1] 이미지 로드 및 리사이징
            # ---------------------------------------------------------
            image = cv2.imread(img_path)
            if image is None: continue
            
            resized_img = cv2.resize(image, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(OUT_IMG_DIR, f"{new_file_name}.jpg"), resized_img)

            # ---------------------------------------------------------
            # [단계 2] JSON 데이터 파싱 및 스케일링
            # ---------------------------------------------------------
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            yolo_lines = []
            # U-Net 마스크 캔버스 초기화 (0: 배경)
            mask_canvas = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
            
            # 디버그용 오버레이 캔버스 초기화
            debug_img = resized_img.copy()
            color_mask = np.zeros_like(resized_img, dtype=np.uint8)

            # 횡단보도를 먼저 그리고 차선을 나중에 그리기 위해 분리
            crosswalk_shapes = []
            lane_shapes = []

            for shape in data['shapes']:
                label = shape['label']
                points = shape['points']
                
                # 좌표 스케일링 (1920x1080 -> 640x360)
                scaled_points = [[p[0] * SCALE_X, p[1] * SCALE_Y] for p in points]
                pts_array = np.array(scaled_points, np.int32)

                # --- YOLO 바운딩 박스 처리 ---
                if label in YOLO_CLASSES:
                    class_id = YOLO_CLASSES[label]
                    x_coords = [p[0] for p in scaled_points]
                    y_coords = [p[1] for p in scaled_points]
                    
                    x_min, x_max = max(0, min(x_coords)), min(TARGET_W, max(x_coords))
                    y_min, y_max = max(0, min(y_coords)), min(TARGET_H, max(y_coords))
                    
                    # 정규화 연산 (0.0 ~ 1.0)
                    center_x = ((x_min + x_max) / 2.0) / TARGET_W
                    center_y = ((y_min + y_max) / 2.0) / TARGET_H
                    box_w = (x_max - x_min) / TARGET_W
                    box_h = (y_max - y_min) / TARGET_H
                    
                    yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}")
                    
                    # 디버그 시각화 (YOLO)
                    cv2.rectangle(debug_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 255), 2)
                    cv2.putText(debug_img, label, (int(x_min), int(y_min)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # --- U-Net 분리 ---
                elif label == 'crosswalk':
                    crosswalk_shapes.append(pts_array)
                elif label == 'lane':
                    lane_shapes.append(pts_array)
                else:
                    # 'd'와 같은 오타 라벨은 자연스럽게 무시됩니다.
                    pass 

            # ---------------------------------------------------------
            # [단계 3] U-Net 마스크 렌더링 (순서가 매우 중요함)
            # ---------------------------------------------------------
            # 1순위: 횡단보도를 먼저 바닥에 깔아줍니다 (초록색)
            for pts in crosswalk_shapes:
                x, y, w, h = cv2.boundingRect(pts)
                # 정답 마스크에 Class 2 기록
                cv2.rectangle(mask_canvas, (x, y), (x+w, y+h), UNET_CLASSES['crosswalk'], -1)
                # 시각화 마스크에 초록색 기록
                cv2.rectangle(color_mask, (x, y), (x+w, y+h), (0, 255, 0), -1)

            # 2순위: 차선을 그 위에 덮어 그립니다 (빨간색)
            for pts in lane_shapes:
                pts = pts.reshape((-1, 1, 2))
                # 정답 마스크에 Class 1 기록 (횡단보도 픽셀을 덮어씀)
                cv2.polylines(mask_canvas, [pts], isClosed=False, color=UNET_CLASSES['lane'], thickness=LANE_THICKNESS)
                # 시각화 마스크에 빨간색 기록
                cv2.polylines(color_mask, [pts], isClosed=False, color=(0, 0, 255), thickness=LANE_THICKNESS)

            # ---------------------------------------------------------
            # [단계 4] 데이터 파일 저장
            # ---------------------------------------------------------
            # YOLO txt 저장
            if yolo_lines:
                with open(os.path.join(OUT_LBL_DIR, f"{new_file_name}.txt"), "w") as f:
                    f.write("\n".join(yolo_lines))

            # U-Net png 저장 (정수형 마스크)
            cv2.imwrite(os.path.join(OUT_MSK_DIR, f"{new_file_name}.png"), mask_canvas)

            # 디버그 이미지 합성 및 저장 (원본 + 마스크 반투명 오버레이)
            overlay_debug = cv2.addWeighted(debug_img, 0.7, color_mask, 0.3, 0)
            cv2.imwrite(os.path.join(OUT_DBG_DIR, f"{new_file_name}.jpg"), overlay_debug)

            total_processed += 1

    print(f"✅ 변환 완료! 총 {total_processed}개의 현실 데이터셋이 성공적으로 구축되었습니다.")
    print(f"📂 저장 경로: {OUTPUT_BASE}")

if __name__ == "__main__":
    main()

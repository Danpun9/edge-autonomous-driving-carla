"""
File: step23_aihub_to_dataset.py

Purpose:
    Convert AI-Hub road-lane annotations into the project's evaluation format.

Main Responsibilities:
    - Read AI-Hub image and JSON annotation pairs.
    - Select ego-lane-like traffic lane annotations with geometry filters.
    - Save resized images, binary masks, and debug overlays under _dataset_aihub_eval/.

Notes:
    The generated evaluation dataset is large and excluded from Git.
"""

import os
import json
import cv2
import math
import numpy as np
from tqdm import tqdm

from src import config as project_config

# ==========================================
# 1. 경로 및 설정
# ==========================================
AIHUB_BASE = project_config.DATASET_AIHUB_DIR
IMG_IN_DIR = os.path.join(AIHUB_BASE, "origin", "1900_1200", "daylight")
JSON_IN_DIR = os.path.join(AIHUB_BASE, "label", "1900_1200", "daylight")

OUTPUT_BASE = project_config.DATASET_AIHUB_EVAL_DIR
OUT_IMG_DIR = os.path.join(OUTPUT_BASE, "images")
OUT_MSK_DIR = os.path.join(OUTPUT_BASE, "masks")
OUT_DBG_DIR = os.path.join(OUTPUT_BASE, "debug")

ORIGINAL_W, ORIGINAL_H = 1920, 1200
TARGET_W, TARGET_H = 640, 360

SCALE_X = TARGET_W / ORIGINAL_W
SCALE_Y = TARGET_H / ORIGINAL_H
LANE_THICKNESS = 10 

# ==========================================
# 2. 에고 차선(Ego-lane) 기하학적 필터링 함수
# ==========================================
def analyze_lane_geometry(pts):
    """
    폴리라인 좌표를 분석하여 유효한 차선인지 판별하고, 하단 X-절편을 반환합니다.
    """
    # Y좌표 기준으로 정렬 (위에서 아래로)
    pts_sorted = sorted(pts, key=lambda p: p['y'])
    p_top, p_bot = pts_sorted[0], pts_sorted[-1]
    
    # 1. 각도(Slope) 필터링: 수평에 가까운 교차로 선 제거
    dx = p_bot['x'] - p_top['x']
    dy = p_bot['y'] - p_top['y']
    if dy == 0: return False, 0
    
    angle = math.degrees(math.atan2(dy, dx)) # 0 ~ 180도
    if not (30 < angle < 150):
        return False, 0
        
    # 2. ROI 필터링: 화면 하위 40%까지 내려오지 않는 멀리 있는 선 제거
    if p_bot['y'] < ORIGINAL_H * 0.6:
        return False, 0
        
    # 3. 화면 최하단(Y=ORIGINAL_H)에서의 X-절편(Intercept) 계산
    # 가장 아래쪽 두 점의 기울기를 사용하여 바닥에 닿는 위치를 추정
    p1 = pts_sorted[-2] if len(pts_sorted) > 1 else p_top
    p2 = p_bot
    dx_seg = p2['x'] - p1['x']
    dy_seg = p2['y'] - p1['y']
    
    if dy_seg != 0:
        x_intercept = p2['x'] + (ORIGINAL_H - p2['y']) * (dx_seg / dy_seg)
    else:
        x_intercept = p2['x']
        
    return True, x_intercept

# ==========================================
# 3. 메인 파이프라인
# ==========================================
def main():
    for d in [OUT_IMG_DIR, OUT_MSK_DIR, OUT_DBG_DIR]:
        os.makedirs(d, exist_ok=True)

    json_files = [f for f in os.listdir(JSON_IN_DIR) if f.endswith('.json')]
    print(f"🚀 고도화된 AI-Hub 에고 차선 추출 시작 (총 {len(json_files)}개)")

    processed_count = 0
    
    for json_file in tqdm(json_files, desc="Processing"):
        base_name = os.path.splitext(json_file)[0]
        json_path = os.path.join(JSON_IN_DIR, json_file)
        img_path = os.path.join(IMG_IN_DIR, f"{base_name}.jpg")

        if not os.path.exists(img_path): continue

        image = cv2.imread(img_path)
        if image is None: continue
        resized_img = cv2.resize(image, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        mask_canvas = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
        debug_img = resized_img.copy()
        color_mask = np.zeros_like(resized_img, dtype=np.uint8)

        # ---------------------------------------------------------
        # [핵심 로직] 유효한 에고 차선 탐색 및 L/R 분류
        # ---------------------------------------------------------
        valid_lines = []
        for annotation in data.get('annotations', []):
            if annotation['class'] == 'traffic_lane':
                pts = annotation['data']
                if len(pts) < 2: continue
                
                is_valid, x_intercept = analyze_lane_geometry(pts)
                if is_valid:
                    valid_lines.append((x_intercept, pts))

        # 중앙(X=960)을 기준으로 좌우 분리
        center_x = ORIGINAL_W / 2
        left_lines = [item for item in valid_lines if item[0] < center_x]
        right_lines = [item for item in valid_lines if item[0] >= center_x]

        ego_lanes = []
        # 좌측 선 중 중앙에 가장 가까운 선 (X절편이 가장 큰 값)
        if left_lines:
            best_left = max(left_lines, key=lambda x: x[0])
            ego_lanes.append(best_left[1])
        # 우측 선 중 중앙에 가장 가까운 선 (X절편이 가장 작은 값)
        if right_lines:
            best_right = min(right_lines, key=lambda x: x[0])
            ego_lanes.append(best_right[1])

        # ---------------------------------------------------------
        # 추출된 에고 차선만 렌더링
        # ---------------------------------------------------------
        for pts in ego_lanes:
            scaled_points = [[p['x'] * SCALE_X, p['y'] * SCALE_Y] for p in pts]
            pts_array = np.array(scaled_points, np.int32).reshape((-1, 1, 2))
            
            cv2.polylines(mask_canvas, [pts_array], isClosed=False, color=1, thickness=LANE_THICKNESS)
            cv2.polylines(color_mask, [pts_array], isClosed=False, color=(0, 0, 255), thickness=LANE_THICKNESS)

        cv2.imwrite(os.path.join(OUT_IMG_DIR, f"{base_name}.jpg"), resized_img)
        cv2.imwrite(os.path.join(OUT_MSK_DIR, f"{base_name}.png"), mask_canvas)
        
        overlay_debug = cv2.addWeighted(debug_img, 0.7, color_mask, 0.3, 0)
        cv2.imwrite(os.path.join(OUT_DBG_DIR, f"{base_name}.jpg"), overlay_debug)

        processed_count += 1

    print(f"\n✅ 에고 차선 전처리 완료! 총 {processed_count}장이 준비되었습니다.")

if __name__ == "__main__":
    main()

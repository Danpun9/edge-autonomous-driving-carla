"""
File: _dataset_real/vid2img.py

Purpose:
    Extract frames from a real-road video file for manual labeling.

Main Responsibilities:
    - Read a configured AVI video from _dataset_real/.
    - Save individual image frames into a case-specific folder.

Notes:
    This is a small one-off preprocessing utility. Adjust video_path and
    output_dir before running it for a different case.
"""

import cv2
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config as project_config

video_path = os.path.join(project_config.DATASET_REAL_DIR, 'case4.avi')
output_dir = os.path.join(project_config.DATASET_REAL_DIR, 'case4_images')
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = round(cap.get(cv2.CAP_PROP_FPS))
frame_interval = fps * 2  # 1초마다 1프레임 추출

count, saved = 0, 0
while cap.isOpened() and saved < 30: # 100장 추출 시 종료
    ret, frame = cap.read()
    if not ret: break
    
    if count % frame_interval == 0:
        cv2.imwrite(os.path.join(output_dir, f"real_{saved:04d}.jpg"), frame)
        saved += 1
    count += 1
cap.release()
print(f"✅ 총 {saved}장의 프레임 추출 완료!")

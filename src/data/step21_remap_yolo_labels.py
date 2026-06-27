"""
File: step21_remap_yolo_labels.py

Purpose:
    Remap CARLA object-detection labels into the project's reduced YOLO class set.

Main Responsibilities:
    - Read labels from _dataset_yolo/labels_yolo_format/.
    - Convert selected class ids into Vehicle/Walker/TrafficLight ids.
    - Save remapped labels under _dataset_yolo/labels_mapped/.

Notes:
    Run before training YOLO with carla_yolo.yaml if mapped labels are required.
"""

import os
import shutil

from src import config as project_config

# ==========================================
# 1. 경로 설정
# ==========================================
BASE_DIR = project_config.DATASET_YOLO_DIR
INPUT_LBL_DIR = os.path.join(BASE_DIR, "labels_yolo_format")
OUTPUT_LBL_DIR = os.path.join(BASE_DIR, "labels_mapped") # 새롭게 저장될 라벨 폴더

# ==========================================
# 2. 클래스 매핑 로직 (수석 엔지니어님 설계 반영)
# ==========================================
# 기존 -> 변경
# 0(vehicle), 1(bike), 2(motobike) -> 0 (Vehicle)
# 3(traffic_light), 4(traffic_sign) -> 2 (TrafficLight)
CLASS_MAPPING = {
    '0': '0', '1': '0', '2': '0',
    '3': '2', '4': '2'
}

def remap_labels(split_name):
    input_dir = os.path.join(INPUT_LBL_DIR, split_name)
    output_dir = os.path.join(OUTPUT_LBL_DIR, split_name)
    
    if not os.path.exists(input_dir):
        print(f"⚠️ 경로를 찾을 수 없습니다: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    processed_count = 0
    for txt_file in txt_files:
        in_path = os.path.join(input_dir, txt_file)
        out_path = os.path.join(output_dir, txt_file)
        
        mapped_lines = []
        with open(in_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    old_class = parts[0]
                    # 매핑 딕셔너리에 있는 클래스만 변환하여 저장
                    if old_class in CLASS_MAPPING:
                        new_class = CLASS_MAPPING[old_class]
                        new_line = f"{new_class} {' '.join(parts[1:])}"
                        mapped_lines.append(new_line)
        
        # 변환된 라벨이 1개라도 있으면 새 파일로 저장
        if mapped_lines:
            with open(out_path, 'w') as f:
                f.write("\n".join(mapped_lines))
            processed_count += 1

    print(f"✅ [{split_name}] 폴더 변환 완료: {processed_count}개 파일 생성")

def main():
    print("🚀 YOLO Dataset Class Remapping 시작...")
    remap_labels("train")
    remap_labels("test")
    print(f"📁 변환된 라벨 저장 위치: {OUTPUT_LBL_DIR}")

if __name__ == "__main__":
    main()

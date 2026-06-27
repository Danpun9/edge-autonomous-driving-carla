"""
File: _dataset_yolo/visualize_yolo.py

Purpose:
    Visualize YOLO-format labels on sample CARLA dataset images.

Main Responsibilities:
    - Load random images and matching YOLO label files.
    - Draw bounding boxes and class ids for inspection.
    - Save debug images under the dataset debug_output folder.

Notes:
    This utility belongs to the bundled YOLO dataset and writes only diagnostic
    images.
"""

import os
import cv2
import random

def main():
    base_dir = r"c:\Users\joons\Downloads\Carla-Object-Detection-Dataset-master"
    images_dir = os.path.join(base_dir, "images", "train")
    labels_dir = os.path.join(base_dir, "labels_yolo_format", "train")
    output_dir = os.path.join(base_dir, "debug_output")
    
    # 클래스 매핑
    classes = {
        0: 'vehicle',
        1: 'bike',
        2: 'motobike',
        3: 'traffic_light',
        4: 'traffic_sign'
    }
    
    # 색상 매핑 (BGR format for OpenCV)
    colors = {
        0: (0, 255, 0),     # Green
        1: (255, 0, 0),     # Blue
        2: (0, 0, 255),     # Red
        3: (0, 255, 255),   # Yellow
        4: (255, 0, 255)    # Magenta
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 파일 리스트 가져오기
    all_images = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Total images found: {len(all_images)}")
    
    # 랜덤하게 10개의 이미지 선택 (데이터가 많을 수 있으므로 일부만 선별)
    random.seed(42)  # 재현성을 위한 시드 고정
    selected_images = random.sample(all_images, min(10, len(all_images)))
    
    images_with_labels = 0
    
    for img_name in selected_images:
        img_path = os.path.join(images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        h, w, _ = img.shape
        has_label = False
        
        # 라벨 파일이 존재하고 크기가 0보다 큰지 확인
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # YOLO 정규화 좌표를 실제 픽셀 좌표로 변환
                        x1 = int((x_center - width / 2) * w)
                        y1 = int((y_center - height / 2) * h)
                        x2 = int((x_center + width / 2) * w)
                        y2 = int((y_center + height / 2) * h)
                        
                        # 이미지에 바운딩 박스 그리기
                        color = colors.get(class_id, (255, 255, 255))
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        # 클래스 이름 텍스트 표시
                        class_name = classes.get(class_id, "Unknown")
                        label_text = f"{class_name}"
                        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        has_label = True
        
        # 디버깅 결과 저장
        out_path = os.path.join(output_dir, f"debug_{img_name}")
        cv2.imwrite(out_path, img)
        if has_label:
            images_with_labels += 1
            print(f"Saved visualization with labels: {out_path}")
        else:
            print(f"Saved visualization (no objects): {out_path}")
            
    print(f"\nProcessing complete. Checked {len(selected_images)} images.")
    print(f"Found {images_with_labels} images with actual bounding box labels.")
    print(f"Check the '{output_dir}' directory for visualizations.")

if __name__ == '__main__':
    main()

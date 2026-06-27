"""
File: inference_combined.py

Purpose:
    Prototype combined YOLO and UFLDv2 lane-detection inference on video files.

Main Responsibilities:
    - Load a YOLO model and a UFLDv2 parsingNet model.
    - Process video frames and draw object/lane overlays.
    - Save an annotated output video.

Notes:
    This script references external UFLDv2 modules and culane_res34.pth, which
    are not part of the current project root. Treat it as an experimental
    prototype unless those dependencies are restored.
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as transforms

from src import config as project_config

# UFLDv2 관련 모듈 import (UFLDv2 레포지토리 구조에 맞게 경로 설정 필요)
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print

def init_models():
    """ YOLOv8과 UFLDv2 모델 초기화 """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. YOLOv8 로드
    yolo_model = YOLO(project_config.YOLO_MODEL_PATH)
    yolo_model.to(device)

    # 2. UFLDv2 로드 (CULane, ResNet34 설정 기준)
    # 실제 UFLDv2의 config 파일을 로드하는 과정이 필요합니다.
    # cfg = merge_config('configs/culane_res34.py') 
    # 아래는 구조적 예시입니다.
    cls_num_per_lane = 18 
    ufld_model = parsingNet(pretrained=False, backbone='34', cls_dim=(200, cls_num_per_lane, 4), use_aux=False).to(device)
    
    # 다운받은 가중치 로드
    state_dict = torch.load(project_config.UFLD_CULANE_CHECKPOINT, map_location=device)['model']
    ufld_model.load_state_dict(state_dict, strict=False)
    ufld_model.eval()
    
    return yolo_model, ufld_model, device

def process_video(video_path, output_path):
    yolo_model, ufld_model, device = init_models()
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 영상 저장용 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # UFLDv2용 이미지 전처리
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)), # CULane 기본 사이즈
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    print(f"[{video_path}] 추론을 시작합니다...")
    
    while cap.isOpened():
        ret, frame = cap.isOpened(), cap.read()[1]
        if not ret:
            break
            
        # ---------------------------------------------------------
        # 1. YOLOv8 추론 및 시각화 (원본 프레임 위에 바로 그리기)
        # ---------------------------------------------------------
        yolo_results = yolo_model(frame, verbose=False)
        annotated_frame = yolo_results[0].plot() 
        
        # ---------------------------------------------------------
        # 2. UFLDv2 추론
        # ---------------------------------------------------------
        from PIL import Image
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = img_transforms(img_pil).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            out_j = ufld_model(img_tensor)
            # UFLDv2의 후처리(Post-processing) 로직을 통해 x, y 좌표 리스트 추출
            # (이 부분은 UFLDv2의 demo.py 내 coordinates 추출 로직과 동일하게 작성해야 합니다.)
            # lanes = get_lanes(out_j, ...) 
            
            # 예시용 더미 좌표 (실제 적용 시 추출된 lanes로 대체)
            lanes = [
                [(400, 600), (500, 400), (600, 300)], # 차선 1
                [(800, 600), (700, 400), (650, 300)]  # 차선 2
            ]

        # ---------------------------------------------------------
        # 3. 빨간색 차선 마스킹 (Red Masking)
        # ---------------------------------------------------------
        # OpenCV는 BGR을 사용하므로 빨간색은 (0, 0, 255) 입니다.
        red_color = (0, 0, 255)
        thickness = 8 # 마스킹 두께
        
        for lane in lanes:
            # 좌표를 numpy array로 변환하여 다각형(선형) 렌더링
            pts = np.array(lane, np.int32)
            pts = pts.reshape((-1, 1, 2))
            # 차선을 붉게 덧칠합니다.
            cv2.polylines(annotated_frame, [pts], isClosed=False, color=red_color, thickness=thickness)
            
            # 혹은 반투명한 마스킹을 원하신다면 addWeighted를 활용할 수 있습니다. (아래 팁 참조)

        # 결과 프레임 저장
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"추론 완료! 결과가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    INPUT_VIDEO = project_config.OTHER_SIM_DRIVE_TEST_VIDEO
    OUTPUT_VIDEO = project_config.RESULT_DRIVE_TEST_VIDEO
    process_video(INPUT_VIDEO, OUTPUT_VIDEO)

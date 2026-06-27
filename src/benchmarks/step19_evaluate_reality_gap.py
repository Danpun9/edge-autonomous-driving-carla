"""
File: step19_evaluate_reality_gap.py

Purpose:
    Evaluate Sim2Real performance on processed real-road samples.

Main Responsibilities:
    - Load a trained segmentation model and YOLO detector.
    - Compare predicted lane masks against real-data masks.
    - Estimate object-detection overlap for available labels.

Notes:
    Requires model weights and _dataset_real_processing/. Results are diagnostic
    and depend on the quality of manual labels.
"""

from numpy._core import multiarray
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

from src import config as project_config

# 이전 단계의 U-Net 모듈 임포트
# from src.models.step11_advanced_model import AdvancedUNet
from src.models.step22_smp_model import SMPHybridUNet

# ==========================================
# 1. 경로 및 설정
# ==========================================
DATASET_DIR = project_config.DATASET_REAL_PROCESSED_DIR
IMG_DIR = os.path.join(DATASET_DIR, "images")
MSK_DIR = os.path.join(DATASET_DIR, "masks")
LBL_DIR = os.path.join(DATASET_DIR, "labels")

TARGET_W, TARGET_H = 640, 360
CROP_Y = project_config.CROP_Y # U-Net 입력 시 상단 크롭 영역

YOLO_CLS_MAP = {0: 'Vehicle', 1: 'Walker', 2: 'TrafficLight'}
UNET_CLS_MAP = {1: 'Lane', 2: 'Crosswalk'}

# ==========================================
# 2. 바운딩 박스 IoU 계산 함수 (YOLO 평가용)
# ==========================================
def bb_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# ==========================================
# 3. 메인 평가 루프
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 현실 격차(Reality Gap) 평가 엔진 부팅 중... (Device: {device})")

    # 모델 로드
    # unet_model = AdvancedUNet(in_channels=3, out_channels=3).to(device)
    # unet_model.load_state_dict(torch.load("advanced_best_aug_unet_model.pth", map_location=device)['model_state_dict'])
    # unet_model.eval()

    unet_model = SMPHybridUNet().to(device)
    checkpoint = torch.load(project_config.SMP_RESNET50_CHECKPOINT, map_location=device)
    if 'model_state_dict' in checkpoint:
        unet_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        unet_model.load_state_dict(checkpoint)
    unet_model.eval()

    yolo_model = YOLO(project_config.YOLO_EXPERIMENT_CHECKPOINT)

    # 평가 지표 누적 변수
    unet_inter = {1: 0, 2: 0}
    unet_union = {1: 0, 2: 0}
    
    yolo_metrics = {
        0: {'TP': 0, 'FP': 0, 'FN': 0},
        1: {'TP': 0, 'FP': 0, 'FN': 0},
        2: {'TP': 0, 'FP': 0, 'FN': 0}
    }

    img_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    total_imgs = len(img_files)
    print(f"✅ 총 {total_imgs}장의 현실 데이터셋 평가를 시작합니다.\n")

    for idx, img_name in enumerate(img_files):
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(IMG_DIR, img_name)
        msk_path = os.path.join(MSK_DIR, f"{base_name}.png")
        lbl_path = os.path.join(LBL_DIR, f"{base_name}.txt")

        # ---------------------------------------------------------
        # [Phase A] U-Net mIoU 평가
        # ---------------------------------------------------------
        image = cv2.imread(img_path)
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

        # U-Net 입력 및 정답지 동일하게 크롭 (180픽셀 하단만 평가)
        img_cropped = frame_rgb[CROP_Y:, :, :].astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_cropped).permute(2, 0, 1).unsqueeze(0).to(device)
        gt_cropped = gt_mask[CROP_Y:, :]

        with torch.no_grad():
            outputs = unet_model(img_tensor)
            pred_mask = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()

        # 픽셀 단위 교집합/합집합 계산
        for cls_idx in [1, 2]:
            pred_inds = (pred_mask == cls_idx)
            gt_inds = (gt_cropped == cls_idx)
            
            # 차원 불일치(예: (180,640)과 (180,640,1)) 방지를 위해 squeeze 적용
            pred_inds = np.squeeze(pred_inds)
            gt_inds = np.squeeze(gt_inds)
            
            unet_inter[cls_idx] += np.logical_and(pred_inds, gt_inds).sum()
            unet_union[cls_idx] += np.logical_or(pred_inds, gt_inds).sum()

        # ---------------------------------------------------------
        # [Phase B] YOLO Precision/Recall 평가
        # ---------------------------------------------------------
        # 1. 정답지(GT) 바운딩 박스 로드
        gt_boxes = {0: [], 1: [], 2: []}
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    x1 = int((cx - w/2) * TARGET_W)
                    y1 = int((cy - h/2) * TARGET_H)
                    x2 = int((cx + w/2) * TARGET_W)
                    y2 = int((cy + h/2) * TARGET_H)
                    gt_boxes[cls_id].append([x1, y1, x2, y2, False]) # False: 매칭 여부

        # 2. YOLO 추론 (사전 학습 모델 사용)
        results = yolo_model.predict(source=image, conf=0.3, verbose=False)[0]
        
        # 3. COCO 예측 클래스를 우리의 커스텀 0, 1, 2 클래스로 매핑
        # pred_boxes = {0: [], 1: [], 2: []}
        # for box in results.boxes:
        #     coco_cls = int(box.cls[0])
        #     conf = float(box.conf[0])
        #     x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
        #     my_cls = -1
            
        #     if coco_cls in [2, 3, 5, 7]: my_cls = 0 # Vehicle
        #     elif coco_cls == 0: my_cls = 1          # Walker
        #     elif coco_cls == 9: my_cls = 2          # Traffic Light
            
        #     if my_cls != -1:
        #         pred_boxes[my_cls].append({'box': [x1, y1, x2, y2], 'conf': conf})

        pred_boxes = {0: [], 1: [], 2: []}
        for box in results.boxes:
            my_cls = int(box.cls[0]) # 커스텀 모델은 이미 0, 1, 2로 맞춰져 있으므로 그대로 사용
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            if my_cls in [0, 1, 2]:
                pred_boxes[my_cls].append({'box': [x1, y1, x2, y2], 'conf': conf})

        # 4. IoU 0.5 기준으로 TP, FP, FN 판별
        for cls_id in [0, 1, 2]:
            preds = sorted(pred_boxes[cls_id], key=lambda x: x['conf'], reverse=True)
            gts = gt_boxes[cls_id]
            
            for p in preds:
                best_iou = 0
                best_gt_idx = -1
                for i, gt in enumerate(gts):
                    if not gt[4]: # 아직 매칭 안 된 정답 박스
                        iou = bb_iou(p['box'], gt[:4])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i
                
                if best_iou >= 0.5:
                    yolo_metrics[cls_id]['TP'] += 1
                    gts[best_gt_idx][4] = True # 매칭 처리
                else:
                    yolo_metrics[cls_id]['FP'] += 1
            
            # 매칭되지 못한 정답 박스들은 모두 FN(탐지 실패)
            yolo_metrics[cls_id]['FN'] += sum(1 for gt in gts if not gt[4])

    # ==========================================
    # 4. 최종 리포트 출력
    # ==========================================
    print("\n" + "="*50)
    print("📊 [현실 격차(Reality Gap) 벤치마크 리포트]")
    print("="*50)

    print("\n[1] U-Net 성능 (시맨틱 세그멘테이션)")
    for cls_idx, cls_name in UNET_CLS_MAP.items():
        iou = (unet_inter[cls_idx] / unet_union[cls_idx]) if unet_union[cls_idx] > 0 else 0
        print(f" - {cls_name:<12} mIoU: {iou * 100:.2f}%")

    print("\n[2] YOLOv8 성능 (객체 감지 - IoU@0.5 기준)")
    for cls_idx, cls_name in YOLO_CLS_MAP.items():
        TP = yolo_metrics[cls_idx]['TP']
        FP = yolo_metrics[cls_idx]['FP']
        FN = yolo_metrics[cls_idx]['FN']
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f" - {cls_name:<12} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1-Score: {f1:.2f}")
    
    print("="*50)

if __name__ == "__main__":
    main()

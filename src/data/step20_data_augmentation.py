"""
File: step20_data_augmentation.py

Purpose:
    Generate augmented Sim2Real training datasets for U-Net and YOLO pipelines.

Main Responsibilities:
    - Apply color, blur, noise, rain, and shadow augmentations.
    - Preserve segmentation masks while augmenting U-Net images.
    - Copy YOLO labels for image-only augmentations.

Notes:
    Writes _dataset_augmented_unet/ and _dataset_augmented_yolo/. These outputs
    are intentionally excluded from Git.
"""

import os
import cv2
import numpy as np
import albumentations as A
import shutil

from src import config as project_config

# ==========================================
# 1. 경로 및 설정
# ==========================================
# 원본 데이터셋 경로
UNET_IN_DIR = project_config.DATASET_MULTICLASS_DIR
YOLO_IN_DIR = project_config.DATASET_YOLO_DIR

# 증강되어 저장될 새로운 데이터셋 경로
UNET_OUT_DIR = project_config.DATASET_AUG_UNET_DIR
YOLO_OUT_DIR = project_config.DATASET_AUG_YOLO_DIR

# 원본 1장당 몇 장의 노이즈 이미지를 생성할 것인가? (예: 2배 뻥튀기)
AUGMENT_MULTIPLIER = 2 

# ==========================================
# 2. Albumentations 노이즈 파이프라인 (Sim2Real)
# ==========================================
# Albumentations는 RGB 포맷의 이미지를 요구합니다.
transform = A.Compose([
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5), # 조도 변화
    A.OneOf([
        A.MotionBlur(blur_limit=5, p=1.0),
        A.GaussianBlur(blur_limit=5, p=1.0),
    ], p=0.4), # 초점 흔들림
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3), # 센서 노이즈
    A.RandomRain(blur_value=3, p=0.2),           # 빗방울/비
    A.RandomShadow(p=0.2)                        # 그림자
])

# ==========================================
# 3. U-Net 데이터셋 증강 로직
# ==========================================
def augment_unet():
    print("🚀 [1/2] U-Net 데이터셋 증강 시작...")
    in_img_dir = os.path.join(UNET_IN_DIR, "images")
    in_msk_dir = os.path.join(UNET_IN_DIR, "masks")
    out_img_dir = os.path.join(UNET_OUT_DIR, "images")
    out_msk_dir = os.path.join(UNET_OUT_DIR, "masks")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)

    if not os.path.exists(in_img_dir):
        print("U-Net 원본 폴더가 없습니다. 건너뜁니다.")
        return

    images = os.listdir(in_img_dir)
    processed = 0

    for img_name in images:
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(in_img_dir, img_name)
        msk_path = os.path.join(in_msk_dir, f"{base_name}.png")

        if not os.path.exists(msk_path): continue

        image = cv2.imread(img_path)
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        
        # BGR -> RGB 변환 (Albumentations 요구사항)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. 원본 복사 저장
        cv2.imwrite(os.path.join(out_img_dir, f"{base_name}_orig.png"), image)
        cv2.imwrite(os.path.join(out_msk_dir, f"{base_name}_orig.png"), mask)

        # 2. 증강 이미지 생성
        for i in range(AUGMENT_MULTIPLIER):
            augmented = transform(image=image_rgb, mask=mask)
            aug_img_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
            aug_mask = augmented['mask'] # 픽셀 변환이므로 마스크는 그대로임

            cv2.imwrite(os.path.join(out_img_dir, f"{base_name}_aug_{i}.png"), aug_img_bgr)
            cv2.imwrite(os.path.join(out_msk_dir, f"{base_name}_aug_{i}.png"), aug_mask)
        
        processed += 1

    print(f"✅ U-Net 증강 완료: 원본 {processed}장 -> 총 {processed * (AUGMENT_MULTIPLIER + 1)}장으로 확장")

# ==========================================
# 4. YOLO 데이터셋 증강 로직
# ==========================================
def augment_yolo():
    print("🚀 [2/2] YOLO 데이터셋 증강 시작...")
    # Train 폴더만 증강합니다. (Test는 오염시키지 않거나 원본 그대로 씁니다)
    in_img_dir = os.path.join(YOLO_IN_DIR, "images", "train")
    in_lbl_dir = os.path.join(YOLO_IN_DIR, "labels_mapped", "train")
    
    out_img_dir = os.path.join(YOLO_OUT_DIR, "images", "train")
    out_lbl_dir = os.path.join(YOLO_OUT_DIR, "labels", "train")
    
    # YOLO validation 폴더 구조도 맞춰줍니다 (증강 없이 원본만 복사)
    out_val_img = os.path.join(YOLO_OUT_DIR, "images", "val")
    out_val_lbl = os.path.join(YOLO_OUT_DIR, "labels", "val")

    for d in [out_img_dir, out_lbl_dir, out_val_img, out_val_lbl]:
        os.makedirs(d, exist_ok=True)

    # 1. Train 증강 처리
    if os.path.exists(in_img_dir):
        images = [f for f in os.listdir(in_img_dir) if f.endswith(('.png', '.jpg'))]
        processed = 0

        for img_name in images:
            base_name = os.path.splitext(img_name)[0]
            img_path = os.path.join(in_img_dir, img_name)
            lbl_path = os.path.join(in_lbl_dir, f"{base_name}.txt")

            if not os.path.exists(lbl_path): continue

            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 원본 저장
            cv2.imwrite(os.path.join(out_img_dir, f"{base_name}_orig.jpg"), image)
            shutil.copy(lbl_path, os.path.join(out_lbl_dir, f"{base_name}_orig.txt"))

            # 증강 저장 (픽셀 변화만 있으므로 라벨 좌표(txt)는 원본과 100% 동일하게 복사)
            for i in range(AUGMENT_MULTIPLIER):
                augmented = transform(image=image_rgb)
                aug_img_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(os.path.join(out_img_dir, f"{base_name}_aug_{i}.jpg"), aug_img_bgr)
                shutil.copy(lbl_path, os.path.join(out_lbl_dir, f"{base_name}_aug_{i}.txt"))
            
            processed += 1
        print(f"✅ YOLO Train 증강 완료: 원본 {processed}장 -> 총 {processed * (AUGMENT_MULTIPLIER + 1)}장으로 확장")

    # 2. Test(Val) 폴더는 원본 그대로 복사 (평가용이므로 노이즈 주입 X)
    in_val_img = os.path.join(YOLO_IN_DIR, "images", "test")
    in_val_lbl = os.path.join(YOLO_IN_DIR, "labels_mapped", "test")
    if os.path.exists(in_val_img) and os.path.exists(in_val_lbl):
        for f in os.listdir(in_val_img): shutil.copy(os.path.join(in_val_img, f), os.path.join(out_val_img, f))
        for f in os.listdir(in_val_lbl): shutil.copy(os.path.join(in_val_lbl, f), os.path.join(out_val_lbl, f))
        print("✅ YOLO Val 데이터 복사 완료")

def main():
    print("==================================================")
    print("🌪️ Sim2Real Data Augmentation Engine (Albumentations)")
    print("==================================================")
    augment_unet()
    augment_yolo()
    print("\n🎉 모든 데이터셋의 증강이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    main()

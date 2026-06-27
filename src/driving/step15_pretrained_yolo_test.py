"""
File: step15_pretrained_yolo_test.py

Purpose:
    Test a pretrained Ultralytics YOLO model inside a CARLA camera loop.

Main Responsibilities:
    - Connect to CARLA and stream front camera frames.
    - Load yolov8n.pt through the Ultralytics API.
    - Display object detection overlays for quick validation.

Notes:
    Requires CARLA, OpenCV display support, and a YOLO weights file or network
    access for automatic download.
"""

import carla
import numpy as np
import cv2
import queue
from ultralytics import YOLO

from src import config as project_config

# ==========================================
# 1. 하이퍼파라미터 및 설정
# ==========================================
IM_WIDTH = project_config.IMAGE_WIDTH
IM_HEIGHT = project_config.IMAGE_HEIGHT
FOV = project_config.CAMERA_FOV

# COCO 데이터셋에서 우리가 자율주행에 필요한 클래스만 필터링
# 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck, 9: traffic light
TARGET_CLASSES = [0, 2, 3, 5, 7, 9]

def main():
    print("🚀 Pre-trained YOLOv8 모델 로드 중...")
    # 'yolov8n.pt' 파일이 없으면 인터넷에서 자동으로 다운로드 받습니다.
    yolo_model = YOLO(project_config.YOLO_MODEL_PATH)
    print("✅ YOLOv8 로드 완료!")

    client = carla.Client(project_config.CARLA_HOST, project_config.CARLA_PORT)
    client.set_timeout(project_config.CARLA_TIMEOUT_SECONDS)
    world = client.get_world()
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    try:
        blueprint_library = world.get_blueprint_library()
        
        # 내 차량 생성
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)

        # 카메라 생성
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        camera_bp.set_attribute('fov', str(FOV))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-10, yaw=0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        print(f"=== Step 15: YOLOv8 Zero-Shot Inference Started ===")

        while True:
            world.tick()
            image = image_queue.get()

            # 이미지 전처리
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape((IM_HEIGHT, IM_WIDTH, 4))
            frame_bgr = array[:, :, :3]

            # ==========================================
            # [핵심] YOLO 실시간 추론 (학습 없이 바로 사용!)
            # ==========================================
            # conf=0.4: 신뢰도가 40% 이상인 객체만 탐지
            # classes: 사람, 차, 신호등 등 지정한 클래스만 탐지
            results = yolo_model.predict(source=frame_bgr, conf=0.4, classes=TARGET_CLASSES, verbose=False)
            
            # YOLO가 제공하는 내장 시각화 함수로 프레임에 박스 그리기
            annotated_frame = results[0].plot()

            # 결과 출력
            cv2.imshow("Pre-trained YOLOv8 Vision", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("종료합니다.")
        settings.synchronous_mode = False
        world.apply_settings(settings)
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

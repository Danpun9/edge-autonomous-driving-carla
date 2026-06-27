"""
File: step09_dl_data_collector.py

Purpose:
    Collect binary lane-segmentation training data from CARLA.

Main Responsibilities:
    - Capture synchronized RGB and semantic-segmentation frames.
    - Convert CARLA semantic labels into lane masks.
    - Save image/mask pairs under _dataset/.

Notes:
    Requires a running CARLA server. The generated dataset is large and is
    excluded by .gitignore.
"""

import carla
import numpy as np
import cv2
import os
import queue
import random

from src import config as project_config

# ==========================================
# 1. 하이퍼파라미터 및 데이터셋 디렉토리 설정
# ==========================================
IM_WIDTH = project_config.IMAGE_WIDTH
IM_HEIGHT = project_config.IMAGE_HEIGHT
FOV = project_config.CAMERA_FOV
SAVE_INTERVAL = 5  # 몇 틱(Tick)마다 이미지를 저장할 것인가? (5 = 1초에 약 4장 저장)

# 저장 폴더 생성 (없으면 자동 생성)
DATASET_DIR = project_config.DATASET_BINARY_DIR
IMG_DIR = os.path.join(DATASET_DIR, "images")
MASK_DIR = os.path.join(DATASET_DIR, "masks")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

def main():
    client = carla.Client(project_config.CARLA_HOST, project_config.CARLA_PORT)
    client.set_timeout(project_config.CARLA_TIMEOUT_SECONDS)
    world = client.get_world()
    
    # 동기 모드 설정 (RGB와 Segmentation의 완벽한 짝을 맞추기 위해 필수)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    try:
        blueprint_library = world.get_blueprint_library()
        
        # 1. 차량 스폰 및 다양한 환경 주행을 위한 오토파일럿 활성화
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        # 4. 다양한 스폰 지점 활용 (외곽 루프를 피해 랜덤으로 설정)
        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)
        
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        
        # 1. Traffic Manager의 시드(Seed) 무작위 변경
        traffic_manager.set_random_device_seed(random.randint(0, int(1e6)))
        
        # 2. 강제 루트 지정 및 3. 차선 변경/신호 무시 확률 조정
        # 외곽 도로를 벗어나 시내로 진입하게끔 유도
        traffic_manager.set_route(vehicle, ["Left", "Right", "Straight", "Left", "Right"])
        traffic_manager.random_left_lanechange_percentage(vehicle, 50.0)
        traffic_manager.random_right_lanechange_percentage(vehicle, 50.0)
        traffic_manager.ignore_lights_percentage(vehicle, 30.0)

        # 2. 동일한 위치에 2개의 카메라(RGB, Semantic) 스폰
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-10, yaw=0))
        
        # 2-A. RGB 카메라 (Input X)
        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(IM_WIDTH))
        rgb_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        rgb_bp.set_attribute('fov', str(FOV))
        rgb_camera = world.spawn_actor(rgb_bp, camera_transform, attach_to=vehicle)
        rgb_queue = queue.Queue()
        rgb_camera.listen(rgb_queue.put)

        # 2-B. Semantic Segmentation 카메라 (Label Y)
        seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(IM_WIDTH))
        seg_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        seg_bp.set_attribute('fov', str(FOV))
        seg_camera = world.spawn_actor(seg_bp, camera_transform, attach_to=vehicle)
        seg_queue = queue.Queue()
        seg_camera.listen(seg_queue.put)

        print(f"=== Step 09: Deep Learning Data Collection Started ===")
        print(f"Data will be saved in: {os.path.abspath(DATASET_DIR)}")

        frame_count = 0
        saved_count = 0

        while True:
            world.tick()
            frame_count += 1
            
            # 동기화된 큐에서 데이터를 가져옴
            rgb_image = rgb_queue.get()
            seg_image = seg_queue.get()

            # 5. 주행 경로 시각화 및 디버깅 (현재 위치 위주)
            waypoint = world.get_map().get_waypoint(vehicle.get_location())
            world.debug.draw_point(waypoint.transform.location, size=0.1, color=carla.Color(255,0,0), life_time=0.1)

            # 저장 인터벌(SAVE_INTERVAL)에 맞춰 데이터 추출
            if frame_count % SAVE_INTERVAL == 0:
                # ==========================================
                # 3. 데이터 처리 및 정답지(Label) 추출
                # ==========================================
                # RGB 이미지 처리
                rgb_array = np.frombuffer(rgb_image.raw_data, dtype=np.dtype("uint8"))
                rgb_array = np.reshape(rgb_array, (IM_HEIGHT, IM_WIDTH, 4))
                rgb_frame = rgb_array[:, :, :3]

                # Semantic Segmentation 이미지 처리
                seg_array = np.frombuffer(seg_image.raw_data, dtype=np.dtype("uint8"))
                seg_array = np.reshape(seg_array, (IM_HEIGHT, IM_WIDTH, 4))
                
                # CARLA는 Class ID를 Red 채널(OpenCV에서는 BGR이므로 인덱스 2)에 저장합니다.
                class_ids = seg_array[:, :, 2]
                
                # [핵심] 차선(RoadLines, ID=6)만 하얀색(255)으로 추출하여 이진 마스크 생성
                lane_mask = np.zeros_like(class_ids)
                lane_mask[class_ids == 24] = 255

                # ==========================================
                # 4. 데이터셋 저장 (Save to Disk)
                # ==========================================
                file_name = f"{saved_count:06d}.png"
                
                cv2.imwrite(os.path.join(IMG_DIR, file_name), rgb_frame)
                cv2.imwrite(os.path.join(MASK_DIR, file_name), lane_mask)
                
                saved_count += 1
                
                # 시각적 확인을 위해 화면에 출력
                cv2.imshow("1. Input (RGB)", rgb_frame)
                cv2.imshow("2. Ground Truth (Lane Mask)", lane_mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n데이터 수집 종료. 총 {saved_count}장의 데이터셋이 생성되었습니다.")
        settings.synchronous_mode = False
        world.apply_settings(settings)
        rgb_camera.destroy()
        seg_camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

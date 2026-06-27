"""
File: step09_advanced_collector.py

Purpose:
    Collect multi-class segmentation data from CARLA for lane and crosswalk
    perception.

Main Responsibilities:
    - Capture RGB and semantic frames in synchronous mode.
    - Project ego-lane boundaries to separate lane pixels from other markings.
    - Save class-index masks under _dataset_multiclass/.

Notes:
    Requires CARLA. Output masks use integer classes where 0 is background, 1
    is lane, and 2 is crosswalk/other road marking.
"""

import carla
import numpy as np
import cv2
import os
import queue
import math
import random

from src import config as project_config

# ==========================================
# 1. 하이퍼파라미터 및 설정
# ==========================================
IM_WIDTH = project_config.IMAGE_WIDTH
IM_HEIGHT = project_config.IMAGE_HEIGHT
FOV = project_config.CAMERA_FOV
SAVE_INTERVAL = 5  
MIN_SPEED_MPS = 1.0 # 정차 중(신호 대기) 중복 데이터 수집 방지용 (1m/s)

DATASET_DIR = project_config.DATASET_MULTICLASS_DIR
IMG_DIR = os.path.join(DATASET_DIR, "images")
MASK_DIR = os.path.join(DATASET_DIR, "masks") # 이제 0, 1, 2 값을 가집니다.
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

# 카메라 Intrinsic Matrix (내부 행렬) 계산
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

# 3D 월드 좌표 -> 2D 이미지 픽셀 변환
def get_image_point(loc, K, w2c):
    # loc은 carla.Location 객체
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    
    # 언리얼 엔진 좌표계(UE4)에서 표준 3D 좌표계로 변환: (x, y, z) -> (y, -z, x)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    
    # 3D -> 2D 투영
    point_img = np.dot(K, point_camera)
    
    # 정규화 (Z값으로 나누기)
    if point_img[2] > 0: # 카메라 앞쪽에 있는 점만
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]
        return int(point_img[0]), int(point_img[1])
    else:
        return None, None

def main():
    client = carla.Client(project_config.CARLA_HOST, project_config.CARLA_PORT)
    client.set_timeout(project_config.CARLA_TIMEOUT_SECONDS)
    world = client.get_world()
    carla_map = world.get_map()
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    try:
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = carla_map.get_spawn_points()
        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        
        # Random behavior
        traffic_manager.set_random_device_seed(random.randint(0, int(1e6)))
        traffic_manager.set_route(vehicle, ["Left", "Right", "Straight", "Left", "Right"])
        traffic_manager.random_left_lanechange_percentage(vehicle, 50.0)
        traffic_manager.random_right_lanechange_percentage(vehicle, 50.0)
        traffic_manager.ignore_lights_percentage(vehicle, 30.0)

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-10, yaw=0))
        
        # RGB 카메라
        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(IM_WIDTH))
        rgb_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        rgb_bp.set_attribute('fov', str(FOV))
        rgb_camera = world.spawn_actor(rgb_bp, camera_transform, attach_to=vehicle)
        rgb_queue = queue.Queue()
        rgb_camera.listen(rgb_queue.put)

        # Semantic 카메라
        seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(IM_WIDTH))
        seg_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        seg_bp.set_attribute('fov', str(FOV))
        seg_camera = world.spawn_actor(seg_bp, camera_transform, attach_to=vehicle)
        seg_queue = queue.Queue()
        seg_camera.listen(seg_queue.put)

        # 투영 행렬 준비
        K = build_projection_matrix(IM_WIDTH, IM_HEIGHT, FOV)

        print(f"=== Step 09: Advanced Multi-class Data Engine Started ===")
        
        frame_count = 0
        saved_count = 0

        while True:
            world.tick()
            frame_count += 1
            
            rgb_image = rgb_queue.get()
            seg_image = seg_queue.get()

            # [핵심 1] 차량 속도 체크 (신호 대기 중 데이터 중복 방지)
            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2)
            
            if frame_count % SAVE_INTERVAL == 0 and speed > MIN_SPEED_MPS:
                # RGB 전처리
                rgb_array = np.frombuffer(rgb_image.raw_data, dtype=np.dtype("uint8")).reshape((IM_HEIGHT, IM_WIDTH, 4))
                rgb_frame = rgb_array[:, :, :3]

                # Semantic 전처리 (CARLA 0.9.16 기준 도로 표식은 24번 태그)
                seg_array = np.frombuffer(seg_image.raw_data, dtype=np.dtype("uint8")).reshape((IM_HEIGHT, IM_WIDTH, 4))
                class_ids = seg_array[:, :, 2]
                base_markings = (class_ids == 24) # 차선 + 횡단보도 전체

                # [핵심 2] Waypoint 기반 에고 차선(Ego-lane) 3D 추출 및 2D 투영
                w2c = rgb_camera.get_transform().get_inverse_matrix()
                current_wp = carla_map.get_waypoint(vehicle.get_location())
                
                left_pts_2d, right_pts_2d = [], []
                
                # 전방 0m ~ 40m 까지 2m 간격으로 차선 경계선 추출
                for d in range(0, 40, 2):
                    if d == 0:
                        next_wps = [current_wp]
                    else:
                        next_wps = current_wp.next(float(d))
                    if not next_wps: break
                    wp = next_wps[0]
                    
                    # 차선의 중심에서 좌/우 폭(width)의 절반만큼 이동
                    right_vec = wp.transform.get_right_vector()
                    half_width = wp.lane_width / 2.0
                    
                    left_loc = wp.transform.location - right_vec * half_width
                    right_loc = wp.transform.location + right_vec * half_width
                    
                    lx, ly = get_image_point(left_loc, K, w2c)
                    rx, ry = get_image_point(right_loc, K, w2c)
                    
                    if lx is not None and rx is not None:
                        left_pts_2d.append([lx, ly])
                        right_pts_2d.append([rx, ry])

                # 2D 좌표를 이용해 이미지 캔버스에 아주 두꺼운 다각형 렌더링 (관심 영역)
                drawn_lanes = np.zeros((IM_HEIGHT, IM_WIDTH), dtype=np.uint8)
                if len(left_pts_2d) > 1:
                    pts = np.array(left_pts_2d, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(drawn_lanes, [pts], False, 255, thickness=40) # 차선을 충분히 덮을 두께
                if len(right_pts_2d) > 1:
                    pts = np.array(right_pts_2d, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(drawn_lanes, [pts], False, 255, thickness=40)

                # [핵심 3] 마스크 분리 연산 (Intersection & Subtraction)
                # 1. 24번 태그 중, 우리가 그린 관심 영역(drawn_lanes) 안에 들어오는 픽셀만 '진짜 차선(Class 1)'
                class_1_lanes = base_markings & (drawn_lanes > 0)
                
                # 2. 24번 태그 중, 차선이 아닌 나머지 영역들은 모두 '횡단보도 및 기타(Class 2)'
                class_2_crosswalks = base_markings & ~class_1_lanes

                # 3. 최종 다중 클래스 인덱스 마스크 조합
                final_index_mask = np.zeros((IM_HEIGHT, IM_WIDTH), dtype=np.uint8)
                final_index_mask[class_1_lanes] = 1 # 1: 주행 차선
                final_index_mask[class_2_crosswalks] = 2 # 2: 횡단보도/격자/정지선

                # 데이터 저장
                file_name = f"{saved_count:06d}.png"
                cv2.imwrite(os.path.join(IMG_DIR, file_name), rgb_frame)
                cv2.imwrite(os.path.join(MASK_DIR, file_name), final_index_mask)
                saved_count += 1

                # [디버깅 시각화] 인간의 눈으로 0, 1, 2는 구분이 안 되므로 값에 색상을 입혀 출력
                vis_mask = np.zeros((IM_HEIGHT, IM_WIDTH, 3), dtype=np.uint8)
                vis_mask[final_index_mask == 1] = [0, 0, 255] # 차선은 빨간색
                vis_mask[final_index_mask == 2] = [0, 255, 0] # 횡단보도는 초록색
                
                # 원본 이미지와 50% 섞어서 투명 오버레이
                overlay = cv2.addWeighted(rgb_frame, 0.5, vis_mask, 0.5, 0)
                
                cv2.imshow("1. RGB Camera", rgb_frame)
                cv2.imshow("2. Auto Labeled (Red:Lane, Green:Crosswalk)", overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n데이터 수집 종료. 총 {saved_count}장의 다중 클래스 데이터셋이 생성되었습니다.")
        settings.synchronous_mode = False
        world.apply_settings(settings)
        rgb_camera.destroy()
        seg_camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

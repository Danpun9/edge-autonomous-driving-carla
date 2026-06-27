"""
File: debug_yolo.py

Purpose:
    Debug CARLA-to-YOLO auto-label generation with visual overlays.

Main Responsibilities:
    - Spawn a CARLA ego vehicle, NPC vehicles, and pedestrians.
    - Project 3D actor boxes into camera-space bounding boxes.
    - Save/preview YOLO labels for troubleshooting.

Notes:
    This is a debugging variant of the object-label collection pipeline and
    requires a running CARLA server.
"""

import carla
import numpy as np
import cv2
import os
import queue

from src import config as project_config

# ==========================================
# 1. 하이퍼파라미터 및 디렉토리 설정
# ==========================================
IM_WIDTH = project_config.IMAGE_WIDTH
IM_HEIGHT = project_config.IMAGE_HEIGHT
FOV = project_config.CAMERA_FOV
SAVE_INTERVAL = 10     
MAX_DISTANCE = 50.0    

DATASET_DIR = project_config.DATASET_YOLO_DIR
IMG_DIR = os.path.join(DATASET_DIR, "images")
LBL_DIR = os.path.join(DATASET_DIR, "labels")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

# ==========================================
# 2. 3D -> 2D 수학적 투영 모듈 (Kimbrain Reference)
# ==========================================
def build_projection_matrix(w, h, fov):
    """카메라의 내부 투영 행렬(Intrinsic Matrix) 계산"""
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    """3D 월드 좌표 -> 2D 픽셀 좌표 변환"""
    # 1. 월드 좌표 -> 카메라 좌표 변환
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    
    if np.any(np.isnan(point_camera)):
        return None

    # 2. 언리얼 좌표계(UE4) -> 표준 카메라 좌표계 변환 (y, -z, x)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    
    # 3. 카메라 렌즈 뒤에 있는 점은 투영하지 않음 (Z-Clipping)
    if point_camera[2] <= 0.0:
        return None

    # 4. 카메라 좌표 -> 픽셀 좌표 투영 (Intrinsic Matrix 곱)
    point_img = np.dot(K, point_camera)
    
    if point_img[2] == 0 or np.isnan(point_img[2]):
        return None

    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    
    if np.any(np.isnan(point_img)) or np.any(np.isinf(point_img)):
        return None
        
    return int(point_img[0]), int(point_img[1])

def get_2d_bounding_box(bb, actor_transform, K, w2c):
    """3D 바운딩 박스의 8개 꼭짓점을 2D Min-Max 직사각형으로 변환"""
    vertices = bb.get_world_vertices(actor_transform)
    pts_2d = []
    
    for v in vertices:
        p = get_image_point(v, K, w2c)
        if p is None:
            return None # 꼭짓점 하나라도 카메라 뒤에 있으면 박스 파기
        pts_2d.append(p)
            
    pts_2d = np.array(pts_2d)
    x_min, x_max = np.min(pts_2d[:, 0]), np.max(pts_2d[:, 0])
    y_min, y_max = np.min(pts_2d[:, 1]), np.max(pts_2d[:, 1])
    
    if x_max < 0 or x_min >= IM_WIDTH or y_max < 0 or y_min >= IM_HEIGHT:
        return None
        
    x_min = max(0, int(x_min))
    x_max = min(IM_WIDTH - 1, int(x_max))
    y_min = max(0, int(y_min))
    y_max = min(IM_HEIGHT - 1, int(y_max))
    
    if (x_max - x_min) < 10 or (y_max - y_min) < 10:
        return None
        
    return (x_min, y_min, x_max, y_max)

# ==========================================
# 3. 메인 오토 라벨링 루프
# ==========================================
def main():
    client = carla.Client(project_config.CARLA_HOST, project_config.CARLA_PORT)
    client.set_timeout(project_config.CARLA_TIMEOUT_SECONDS)
    world = client.get_world()
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    try:
        blueprint_library = world.get_blueprint_library()
        
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        camera_bp.set_attribute('fov', str(FOV))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-10, yaw=0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        K = build_projection_matrix(IM_WIDTH, IM_HEIGHT, FOV)

        print("=== Step 15: Re-engineered YOLO Auto Labeling Started ===")
        frame_count = 0
        saved_count = 0

        while frame_count < 200:
            world.tick()
            image = image_queue.get()
            
            snapshot = world.get_snapshot()
            
            camera_snap = snapshot.find(camera.id)
            if not camera_snap: continue
            cam_tf = camera_snap.get_transform()
            w2c = cam_tf.get_inverse_matrix()
            cam_fw_vec = cam_tf.get_forward_vector()

            if frame_count % SAVE_INTERVAL == 0:
                yolo_labels = []

                for actor in world.get_actors().filter('*'):
                    is_vehicle = 'vehicle' in actor.type_id
                    is_walker = 'walker' in actor.type_id
                    is_traffic_light = 'traffic_light' in actor.type_id
                    
                    if not (is_vehicle or is_walker or is_traffic_light): continue
                    if actor.id == vehicle.id: continue

                    actor_snap = snapshot.find(actor.id)
                    if not actor_snap: continue
                    actor_tf = actor_snap.get_transform()

                    if actor_tf.location.distance(cam_tf.location) > MAX_DISTANCE:
                        continue

                    ray_x = actor_tf.location.x - cam_tf.location.x
                    ray_y = actor_tf.location.y - cam_tf.location.y
                    ray_z = actor_tf.location.z - cam_tf.location.z
                    dot_val = cam_fw_vec.x * ray_x + cam_fw_vec.y * ray_y + cam_fw_vec.z * ray_z
                    
                    if dot_val < 2.0: 
                        continue

                    bounding_boxes_to_process = []
                    cls_id = -1
                    
                    if is_vehicle:
                        cls_id = 0
                        bounding_boxes_to_process.append(actor.bounding_box)
                    elif is_walker:
                        cls_id = 1
                        bounding_boxes_to_process.append(actor.bounding_box)
                    elif is_traffic_light:
                        state = actor.get_state()
                        if state in [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow]:
                            cls_id = 2
                        elif state == carla.TrafficLightState.Green:
                            cls_id = 3
                        else:
                            continue
                        bounding_boxes_to_process.extend(actor.get_light_boxes())

                    for bb in bounding_boxes_to_process:
                        bbox_2d = get_2d_bounding_box(bb, actor_tf, K, w2c)
                        if bbox_2d is None: continue
                        
                        yolo_labels.append(f"{cls_id}")

                if len(yolo_labels) > 0:
                    saved_count += 1
                
                print(f"Frame {frame_count}: found {len(yolo_labels)} labels")

            frame_count += 1

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print(f"\n✅ YOLO 데이터 수집 종료. 총 {saved_count}쌍의 데이터셋이 생성되었습니다.")
        settings.synchronous_mode = False
        world.apply_settings(settings)
        camera.destroy()
        vehicle.destroy()

if __name__ == '__main__':
    main()

"""
File: step15_yolo_collector.py

Purpose:
    Collect CARLA object-detection images and YOLO labels automatically.

Main Responsibilities:
    - Spawn ego/NPC vehicles and walkers.
    - Project 3D actor bounding boxes into the camera frame.
    - Save RGB images and normalized YOLO labels under _dataset_yolo/.

Notes:
    Requires CARLA to be running. Generated data is large and excluded from Git.
"""

import carla
import numpy as np
import cv2
import os
import queue
import random

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
    
    # [수정] 점 자체가 NaN인 경우 처리 (드문 경우지만 안전을 위해)
    if np.any(np.isnan(point_camera)):
        return None

    # 2. 언리얼 좌표계(UE4) -> 표준 카메라 좌표계 변환 (y, -z, x)
    # point_camera[0]: UE4 X (Forward), point_camera[1]: UE4 Y (Right), point_camera[2]: UE4 Z (Up)
    # 표준 카메라: X(Right)=Y_ue, Y(Down)=-Z_ue, Z(Forward)=X_ue
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    
    # 3. 카메라 렌즈 뒤에 있는 점은 투영하지 않음 (Z-Clipping)
    if point_camera[2] <= 0.0:
        return None

    # 4. 카메라 좌표 -> 픽셀 좌표 투영 (Intrinsic Matrix 곱)
    point_img = np.dot(K, point_camera)
    
    # [수정] 나누기 전 point_img[2]가 0이거나 NaN인 경우 처리
    if point_img[2] == 0 or np.isnan(point_img[2]):
        return None

    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    
    # [수정] 최종 결과가 NaN이거나 Inf인 경우 정수 변환 시 오류 발생하므로 체크
    if np.any(np.isnan(point_img)) or np.any(np.isinf(point_img)):
        return None
        
    return int(point_img[0]), int(point_img[1])

def get_2d_bounding_box(bb, actor_transform, K, w2c, min_size=10):
    """3D 바운딩 박스의 8개 꼭짓점을 2D Min-Max 직사각형으로 변환"""
    vertices = bb.get_world_vertices(actor_transform)
    pts_2d = []
    
    for v in vertices:
        p = get_image_point(v, K, w2c)
        if p is None:
            return None # 꼭짓점 하나라도 카메라 뒤에 있으면 박스 파기 (화면 덮음 버그 방지)
        pts_2d.append(p)
            
    pts_2d = np.array(pts_2d)
    x_min, x_max = np.min(pts_2d[:, 0]), np.max(pts_2d[:, 0])
    y_min, y_max = np.min(pts_2d[:, 1]), np.max(pts_2d[:, 1])
    
    # 박스가 화면을 완전히 벗어난 경우 무시
    if x_max < 0 or x_min >= IM_WIDTH or y_max < 0 or y_min >= IM_HEIGHT:
        return None
        
    # 화면 밖으로 튀어나간 박스를 화면 테두리에 맞게 자르기 (Clipping)
    x_min = max(0, int(x_min))
    x_max = min(IM_WIDTH - 1, int(x_max))
    y_min = max(0, int(y_min))
    y_max = min(IM_HEIGHT - 1, int(y_max))
    
    # 너무 작거나 찌그러진 박스 무시
    if (x_max - x_min) < min_size or (y_max - y_min) < min_size:
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

    npc_vehicles = []
    npc_walkers = []
    vehicle = None
    camera = None
    saved_count = 0

    try:
        blueprint_library = world.get_blueprint_library()
        
        # 1. 내 차량 생성 및 오토파일럿
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)

        # 1-1. 데이터 수집을 위한 NPC 차량 및 보행자 스폰
        random.shuffle(spawn_points)
        
        # 차량 스폰
        for sp in spawn_points[1:min(31, len(spawn_points))]:
            bp = random.choice(blueprint_library.filter('vehicle.*'))
            npc = world.try_spawn_actor(bp, sp)
            if npc is not None:
                npc.set_autopilot(True)
                npc_vehicles.append(npc)
                
        # 보행자 스폰
        for _ in range(20):
            sp = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                sp.location = loc
                bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
                walker = world.try_spawn_actor(bp, sp)
                if walker is not None:
                    walker_controller_bp = blueprint_library.find('controller.ai.walker')
                    controller = world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
                    controller.start()
                    controller.go_to_location(world.get_random_location_from_navigation())
                    controller.set_max_speed(1 + random.random())
                    npc_walkers.append((walker, controller))
                    
        print(f"Spawned {len(npc_vehicles)} NPC vehicles and {len(npc_walkers)} walkers for YOLO data collection.")

        # 2. RGB 카메라 생성
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

        while True:
            world.tick()
            image = image_queue.get()
            
            # [해결 B] 완벽한 시공간 동기화를 위한 Snapshot 추출 (Lag 방지)
            snapshot = world.get_snapshot()
            
            camera_snap = snapshot.find(camera.id)
            if not camera_snap: continue
            cam_tf = camera_snap.get_transform()
            w2c = cam_tf.get_inverse_matrix()
            cam_fw_vec = cam_tf.get_forward_vector() # 카메라가 바라보는 정면 벡터

            if frame_count % SAVE_INTERVAL == 0:
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape((IM_HEIGHT, IM_WIDTH, 4))
                frame_bgr = array[:, :, :3].copy()
                
                yolo_labels = []

                # 액터들을 순회하며 라벨링
                for actor in world.get_actors().filter('*'):
                    # 차량(0), 보행자(1), 신호등(2,3)만 필터링
                    is_vehicle = 'vehicle' in actor.type_id
                    is_walker = 'walker' in actor.type_id
                    is_traffic_light = 'traffic_light' in actor.type_id
                    
                    if not (is_vehicle or is_walker or is_traffic_light): continue
                    if actor.id == vehicle.id: continue # 내 차는 박스 치지 않음

                    actor_snap = snapshot.find(actor.id)
                    if not actor_snap: continue
                    actor_tf = actor_snap.get_transform()

                    # 1. 거리 필터링
                    if actor_tf.location.distance(cam_tf.location) > MAX_DISTANCE:
                        continue

                    # [해결 A] Kimbrain 내적(Dot Product) 필터링: 카메라 '앞'에 있는지 수학적 검증
                    ray_x = actor_tf.location.x - cam_tf.location.x
                    ray_y = actor_tf.location.y - cam_tf.location.y
                    ray_z = actor_tf.location.z - cam_tf.location.z
                    dot_val = cam_fw_vec.x * ray_x + cam_fw_vec.y * ray_y + cam_fw_vec.z * ray_z
                    
                    if dot_val < 2.0: 
                        continue # 카메라 렌즈보다 2미터 이상 앞에 있지 않으면 발산 위험으로 무시!

                    # 2. 클래스 및 박스 추출 로직
                    bounding_boxes_to_process = []
                    cls_id = -1
                    
                    if is_vehicle:
                        cls_id = 0
                        bounding_boxes_to_process.append(actor.bounding_box)
                    elif is_walker:
                        cls_id = 1
                        bounding_boxes_to_process.append(actor.bounding_box)
                    elif is_traffic_light:
                        # [해결 C] 거대한 트리거 박스 대신 '실제 램프 불빛 박스' 가져오기
                        state = actor.get_state()
                        if state in [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow]:
                            cls_id = 2
                        elif state == carla.TrafficLightState.Green:
                            cls_id = 3
                        else:
                            continue
                        # 해당 신호등 기둥에 달린 여러 개의 전구 박스들을 모두 리스트에 추가
                        bounding_boxes_to_process.extend(actor.get_light_boxes())

                    # 3. 2D 박스 변환 및 YOLO 포맷 저장
                    for bb in bounding_boxes_to_process:
                        min_size = 2 if is_traffic_light else 10
                        bbox_2d = get_2d_bounding_box(bb, actor_tf, K, w2c, min_size=min_size)
                        if bbox_2d is None: continue
                        
                        x_min, y_min, x_max, y_max = bbox_2d

                        # YOLO 포맷 정규화
                        center_x = ((x_min + x_max) / 2.0) / IM_WIDTH
                        center_y = ((y_min + y_max) / 2.0) / IM_HEIGHT
                        width = (x_max - x_min) / IM_WIDTH
                        height = (y_max - y_min) / IM_HEIGHT
                        
                        yolo_labels.append(f"{cls_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

                        # 4. 디버깅 화면 그리기
                        color = (0, 0, 255) if cls_id in [0, 2] else (0, 255, 0)
                        cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), color, 2)
                        cv2.putText(frame_bgr, f"CLS:{cls_id}", (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if len(yolo_labels) > 0:
                    file_name = f"{saved_count:06d}"
                    cv2.imwrite(os.path.join(IMG_DIR, f"{file_name}.png"), frame_bgr)
                    with open(os.path.join(LBL_DIR, f"{file_name}.txt"), "w") as f:
                        f.write("\n".join(yolo_labels))
                    saved_count += 1

                cv2.imshow("YOLO Debugger (Slow Motion)", frame_bgr)

            frame_count += 1
            
            # [해결 3] 디버깅을 위해 OpenCV 화면 갱신을 200ms 대기 (약 5 FPS 체감 속도)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n✅ YOLO 데이터 수집 종료. 총 {saved_count}쌍의 데이터셋이 생성되었습니다.")
        settings.synchronous_mode = False
        world.apply_settings(settings)
        if camera is not None:
            camera.destroy()
        if vehicle is not None:
            vehicle.destroy()
        for npc in npc_vehicles:
            if npc.is_alive:
                npc.destroy()
        for walker, controller in npc_walkers:
            if controller.is_alive:
                controller.stop()
                controller.destroy()
            if walker.is_alive:
                walker.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

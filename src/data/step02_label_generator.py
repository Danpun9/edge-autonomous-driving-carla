"""
File: step02_label_generator.py

Purpose:
    Provide projection helpers for converting CARLA 3D bounding boxes into
    YOLO-style 2D labels.

Main Responsibilities:
    - Build camera intrinsic matrices from resolution and FOV.
    - Project CARLA world coordinates into image coordinates.
    - Format visible vehicle boxes as normalized YOLO labels.

Notes:
    This file is a utility/example module. It depends on live CARLA actors when
    used inside a simulation loop.
"""

import numpy as np
import carla
import os

from src import config as project_config

# ==========================================
# 1. 카메라 설정 (이전 합의대로 640x360)
# ==========================================
IM_WIDTH = project_config.IMAGE_WIDTH
IM_HEIGHT = project_config.IMAGE_HEIGHT
FOV = project_config.CAMERA_FOV

def build_projection_matrix(w, h, fov):
    """카메라의 내부 파라미터(Intrinsic Matrix) K를 계산합니다."""
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_2d_bounding_box(target_vehicle, camera, K_matrix):
    """타겟 차량의 3D Box를 2D 이미지 픽셀로 투영하여 YOLO 포맷으로 반환합니다."""
    # 1. 타겟 차량의 3D Bounding Box 꼭짓점(8개)의 절대 좌표(World Coordinate) 추출
    bb = target_vehicle.bounding_box
    vertices = bb.get_world_vertices(target_vehicle.get_transform())
    
    # 2. 카메라의 외부 파라미터(Extrinsic Matrix) 가져오기 (World -> Camera 역행렬)
    camera_transform = camera.get_transform()
    world_2_camera = np.array(camera_transform.get_inverse_matrix())

    # 좌표 변환을 위한 리스트
    points_2d = []
    
    for vertex in vertices:
        # 3D 점을 동차 좌표계(Homogeneous Coordinates)로 변환: [X, Y, Z, 1]
        vertex_homo = np.array([vertex.x, vertex.y, vertex.z, 1.0])
        
        # World 좌표를 Camera 상대 좌표로 변환
        vertex_cam = np.dot(world_2_camera, vertex_homo)
        
        # CARLA 좌표계(X:앞, Y:오른쪽, Z:위) -> 광학 좌표계(Z:앞, X:오른쪽, Y:아래) 스와핑
        point_cam = np.array([vertex_cam[1], -vertex_cam[2], vertex_cam[0]])
        
        # 카메라 뒤쪽(Z <= 0)에 있는 점은 투영하지 않음
        if point_cam[2] <= 0.0:
            continue
            
        # 3. 2D 이미지 평면으로 투영 (Intrinsic Matrix 곱셈)
        point_img = np.dot(K_matrix, point_cam)
        
        # Z값으로 나누어 정규화 (Perspective Divide)
        u = point_img[0] / point_img[2]
        v = point_img[1] / point_img[2]
        
        points_2d.append([u, v])

    # 8개의 점 중 카메라 앞쪽에 있는 점이 없다면 화면에 안 보이는 것
    if len(points_2d) == 0:
        return None

    points_2d = np.array(points_2d)
    
    # 4. 2D Bounding Box의 Min/Max 좌표 추출 및 화면 밖으로 나가는 부분 잘라내기(Clip)
    min_x = np.clip(np.min(points_2d[:, 0]), 0, IM_WIDTH)
    max_x = np.clip(np.max(points_2d[:, 0]), 0, IM_WIDTH)
    min_y = np.clip(np.min(points_2d[:, 1]), 0, IM_HEIGHT)
    max_y = np.clip(np.max(points_2d[:, 1]), 0, IM_HEIGHT)

    # Box가 너무 작거나 화면을 벗어난 경우 무시 (노이즈 방지)
    if (max_x - min_x) < 5 or (max_y - min_y) < 5:
        return None

    # 5. YOLO 포맷으로 변환 (정규화된 x_center, y_center, width, height)
    x_center = ((min_x + max_x) / 2.0) / IM_WIDTH
    y_center = ((min_y + max_y) / 2.0) / IM_HEIGHT
    width = (max_x - min_x) / IM_WIDTH
    height = (max_y - min_y) / IM_HEIGHT
    
    class_id = 0 # 0: Vehicle (차량 클래스)
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

# 실제 사용 예시 (메인 루프 내부에서 카메라 데이터 수집과 동시에 호출)
# target_vehicles = world.get_actors().filter('vehicle.*')
# K_matrix = build_projection_matrix(IM_WIDTH, IM_HEIGHT, FOV)
# yolo_labels = []
# for vehicle in target_vehicles:
#     if vehicle.id != ego_vehicle.id: # 자기 자신은 제외
#         label = get_2d_bounding_box(vehicle, camera_rgb_center, K_matrix)
#         if label:
#             yolo_labels.append(label)
#
# with open(f"labels/{frame_id:06d}.txt", "w") as f:
#     f.write("\n".join(yolo_labels))

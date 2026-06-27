"""
File: step05_ipm.py

Purpose:
    Demonstrate inverse perspective mapping (IPM) for a CARLA front camera.

Main Responsibilities:
    - Compute perspective-transform matrices for bird's-eye-view conversion.
    - Stream front camera frames from CARLA.
    - Display the original ROI and transformed BEV image.

Notes:
    Requires CARLA and OpenCV display support. It does not write training data.
"""

import carla
import queue
import numpy as np
import cv2

from src import config as project_config

# ==========================================
# 1. 카메라 및 화면 해상도 설정
# ==========================================
IM_WIDTH = project_config.IMAGE_WIDTH
IM_HEIGHT = project_config.IMAGE_HEIGHT
FOV = project_config.CAMERA_FOV

def get_ipm_matrices(w, h):
    """
    원본 이미지의 사다리꼴(ROI)과 목적지의 직사각형 좌표를 매핑하여
    원근 변환(Perspective Transform) 매트릭스를 계산합니다.
    """
    # 1. 원본 이미지에서 도로 영역을 나타내는 사다리꼴의 4개 꼭짓점 (좌하, 좌상, 우상, 우하)
    # ※ 이 값들은 카메라의 높이와 각도(Pitch)에 따라 튜닝해야 합니다.
    src_points = np.float32([
        [40, h],            # 좌측 하단
        [250, h * 0.55],    # 좌측 상단 (소실점 부근)
        [390, h * 0.55],    # 우측 상단
        [600, h]            # 우측 하단
    ])

    # 2. 변환될 BEV(조감도) 이미지의 직사각형 4개 꼭짓점
    # 화면 전체를 채우도록 설정합니다.
    dst_points = np.float32([
        [0, h],             # 좌측 하단
        [0, 0],             # 좌측 상단
        [w, 0],             # 우측 상단
        [w, h]              # 우측 하단
    ])

    # 3. OpenCV를 이용해 3x3 변환 행렬(M)과 역변환 행렬(Minv) 계산
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    
    return M, Minv, src_points

def main():
    client = carla.Client(project_config.CARLA_HOST, project_config.CARLA_PORT)
    client.set_timeout(project_config.CARLA_TIMEOUT_SECONDS)
    world = client.get_world()

    # 동기 모드 설정
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    try:
        blueprint_library = world.get_blueprint_library()
        
        # 차량 스폰 및 자동 주행 활성화 (관찰을 위해)
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True) 
        
        client.get_trafficmanager(8000).set_synchronous_mode(True)

        # ==========================================
        # 2. 카메라 스폰 (도로를 잘 보기 위해 Pitch를 살짝 아래로 내림)
        # ==========================================
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        camera_bp.set_attribute('fov', str(FOV))
        
        # 약간 아래(-10도)를 바라보게 하여 후드의 난반사를 피하고 도로에 집중
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-10, yaw=0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        # 변환 매트릭스 사전 계산 (루프 밖에서 한 번만 연산하여 CPU 절약)
        M, Minv, src_points = get_ipm_matrices(IM_WIDTH, IM_HEIGHT)

        print("=== Step 05: IPM (Bird's Eye View) Started ===")

        while True:
            world.tick()
            image = image_queue.get()

            # CARLA Raw Data -> OpenCV Numpy 포맷 (BGR) 변환
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            frame = array[:, :, :3].copy()

            # ==========================================
            # 3. IPM 변환 수행 및 시각화
            # ==========================================
            # cv2.warpPerspective 연산을 통해 이미지를 조감도로 폅니다.
            bev_image = cv2.warpPerspective(frame, M, (IM_WIDTH, IM_HEIGHT), flags=cv2.INTER_LINEAR)

            # 시각적 디버깅을 위해 원본 이미지에 사다리꼴(ROI) 윤곽선을 그립니다.
            pts = src_points.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # 결과 창 띄우기
            cv2.imshow("1. Original Camera (ROI)", frame)
            cv2.imshow("2. Bird's Eye View (IPM)", bev_image)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("\n종료 및 리소스 정리 중...")
        settings.synchronous_mode = False
        world.apply_settings(settings)
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

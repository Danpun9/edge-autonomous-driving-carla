"""
File: step01_data_collector.py

Purpose:
    Collect synchronized multi-camera perception data from CARLA.

Main Responsibilities:
    - Spawn an ego vehicle and attach RGB/depth sensors.
    - Run CARLA in synchronous mode for frame-aligned sensor capture.
    - Save image streams under _out_perception_data/.

Notes:
    Requires a running CARLA server. Generated sensor data can grow quickly and
    should remain excluded from normal Git commits.
"""

import carla
import queue
import numpy as np
import cv2
import os
import time

from src import config as project_config

# ==========================================
# 1. 하이퍼파라미터 및 센서 설정
# ==========================================
# 해상도: RTX 4080 Super 학습 효율을 위해 800x600 또는 1280x720 권장
IM_WIDTH = project_config.IMAGE_WIDTH
IM_HEIGHT = project_config.PERCEPTION_IMAGE_HEIGHT
FOV = project_config.CAMERA_FOV
FPS = 20  # 초당 20프레임 수집 (물리 엔진 안정성 확보)

SAVE_DIR = project_config.OUT_PERCEPTION_DATA_DIR
SENSOR_NAMES = ["rgb_center", "rgb_left", "rgb_right", "depth_center"]

# 폴더 생성
for name in SENSOR_NAMES:
    os.makedirs(os.path.join(SAVE_DIR, name), exist_ok=True)


def build_camera_blueprint(blueprint_library, sensor_type):
    """센서 블루프린트를 생성하고 해상도/FOV를 설정합니다."""
    bp = blueprint_library.find(sensor_type)
    bp.set_attribute("image_size_x", str(IM_WIDTH))
    bp.set_attribute("image_size_y", str(IM_HEIGHT))
    bp.set_attribute("fov", str(FOV))
    # 센서 틱(Tick)을 시뮬레이터와 동기화
    # bp.set_attribute("sensor_tick", str(1.0 / FPS))  # Disabled in sync mode
    return bp


def main():
    client = carla.Client(project_config.CARLA_HOST, project_config.CARLA_PORT)
    client.set_timeout(project_config.CARLA_TIMEOUT_SECONDS)
    world = client.get_world()

    # ==========================================
    # 2. Synchronous Mode (동기 모드) 활성화
    # ==========================================
    # 실제 차량의 Hardware Clock Sync(PTP)를 모사합니다.
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    # Traffic Manager 동기화 설정 추가
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    vehicle = None
    sensor_list = []

    try:
        blueprint_library = world.get_blueprint_library()

        # Ego Vehicle (테슬라 모델 3 등) 스폰
        vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)  # 데이터 수집을 위해 CARLA 기본 오토파일럿 활성화

        # ==========================================
        # 3. 센서 스폰 및 차량 부착 (Extrinsic Calibration)
        # ==========================================
        # X: 전진, Y: 우측, Z: 위측 (단위: 미터)
        transforms = {
            "rgb_center": carla.Transform(
                carla.Location(x=1.5, z=2.4), carla.Rotation(yaw=0)
            ),
            "rgb_left": carla.Transform(
                carla.Location(x=1.5, z=2.4), carla.Rotation(yaw=-45)
            ),
            "rgb_right": carla.Transform(
                carla.Location(x=1.5, z=2.4), carla.Rotation(yaw=45)
            ),
            "depth_center": carla.Transform(
                carla.Location(x=1.5, z=2.4), carla.Rotation(yaw=0)
            ),
        }

        # 센서 블루프린트 가져오기
        rgb_bp = build_camera_blueprint(blueprint_library, "sensor.camera.rgb")
        depth_bp = build_camera_blueprint(blueprint_library, "sensor.camera.depth")

        # 센서들을 담을 동기화 큐 생성
        sensor_queue = queue.Queue()

        # 콜백 함수: 센서 데이터가 들어오면 큐에 넣음 (센서 이름표 부착)
        def sensor_callback(data, sensor_name):
            sensor_queue.put((sensor_name, data))

        # 4개의 센서 생성 및 부착
        for name, transform in transforms.items():
            bp = depth_bp if "depth" in name else rgb_bp
            sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
            sensor.listen(lambda data, n=name: sensor_callback(data, n))
            sensor_list.append(sensor)

        print("물리 엔진 및 센서 초기화 웜업 중...")
        for _ in range(10):
            world.tick()
            
        # 웜업 동안 큐에 무작위로 쌓인 쓰레기 데이터 비우기 (Sync 맞추기 위해)
        while not sensor_queue.empty():
            sensor_queue.get()

        print("데이터 수집을 시작합니다... (종료: Ctrl+C)")

        # ==========================================
        # 4. Main Data Loop (데이터 수집 및 동기화 루프)
        # ==========================================
        frame_id = 0
        while True:
            # 시뮬레이터 1틱(Tick) 전진
            world.tick()
            frame_id += 1

            # 4개의 센서 데이터가 모두 도착할 때까지 대기 및 분류
            sync_data = {}
            try:
                for _ in range(len(transforms)):
                    s_name, s_data = sensor_queue.get(True, 2.0)
                    sync_data[s_name] = s_data
            except queue.Empty:
                print(f"[{frame_id}] 센서 데이터 수신 타임아웃. 프레임을 건너뜁니다.")
                continue # 프로그램 종료 없이 다음 틱으로 진행

            # 데이터 저장 로직 (Numpy/OpenCV 활용)
            for name, data in sync_data.items():
                # CARLA 원시 데이터를 numpy 배열로 변환
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))  # BGRA 포맷

                if "rgb" in name:
                    img = array[:, :, :3]  # BGR 채널만 추출 (OpenCV 호환)
                    cv2.imwrite(f"{SAVE_DIR}/{name}/{frame_id:06d}.png", img)
                elif "depth" in name:
                    # Depth 이미지는 거리 계산을 위해 별도 변환 (저장 생략 또는 정규화 저장 가능)
                    # 여기서는 시각화용으로 간단히 저장
                    img = array[:, :, :3]
                    cv2.imwrite(f"{SAVE_DIR}/{name}/{frame_id:06d}.png", img)

            # 콘솔 피드백
            if frame_id % 20 == 0:
                print(f"[{frame_id}] 프레임 저장 완료...")

    except KeyboardInterrupt:
        print("\n데이터 수집을 종료합니다.")
    finally:
        # 종료 시 리소스 정리 (매우 중요: 메모리 누수 방지)
        print("센서 및 차량을 파괴합니다...")
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for sensor in sensor_list:
            sensor.destroy()
        if vehicle:
            vehicle.destroy()


if __name__ == "__main__":
    main()

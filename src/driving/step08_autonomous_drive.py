"""
File: step08_autonomous_drive.py

Purpose:
    Run an end-to-end classical lane-following loop in CARLA.

Main Responsibilities:
    - Detect lanes using IPM, binary masking, and polynomial fitting.
    - Convert image-space lane curves into a vehicle-frame/global path.
    - Use VehicleController to steer and control speed.

Notes:
    Requires CARLA to be running. This is an interactive simulation demo rather
    than a standalone unit-test target.
"""

import carla
import queue
import numpy as np
import cv2
import math

from src import config as project_config

# 이전에 만든 핵심 모듈들 임포트
from src.driving.step04_controller import VehicleController
from src.driving.step07_polyfit import get_ipm_matrices, extract_lane_mask, fit_polynomial, LaneTracker, IM_WIDTH, IM_HEIGHT

# 픽셀을 미터로 변환하기 위한 상수 (Calibration)
YM_PER_PIX = 30.0 / 360  # 세로 360픽셀 = 실제 도로 30미터
XM_PER_PIX = 3.5 / 450   # 가로 450픽셀 = 실제 차선폭 3.5미터

def pixel_to_global_path(best_left, best_right, ego_transform):
    """
    비전 모듈이 뽑아낸 좌/우 차선 다항식을 바탕으로 
    실제 제어기가 추종할 글로벌 (x, y, yaw) 궤적을 생성합니다.
    """
    path_x, path_y, path_yaw = [], [], []
    
    if best_left is None and best_right is None:
        return path_x, path_y, path_yaw # 궤적 없음 (비상 정지해야 함)

    # 이미지의 아래쪽(차량 바로 앞)부터 위쪽(전방 30m)으로 궤적 점 생성
    ploty = np.linspace(IM_HEIGHT - 1, 0, num=20)
    
    # 1. 중심선(Center Line) 픽셀 좌표 계산
    if best_left is not None and best_right is not None:
        left_fitx = best_left[0]*ploty**2 + best_left[1]*ploty + best_left[2]
        right_fitx = best_right[0]*ploty**2 + best_right[1]*ploty + best_right[2]
        center_fitx = (left_fitx + right_fitx) / 2.0
    elif best_left is not None:
        left_fitx = best_left[0]*ploty**2 + best_left[1]*ploty + best_left[2]
        center_fitx = left_fitx + (450 / 2) # 차선 절반만큼 우측으로 이동
    else:
        right_fitx = best_right[0]*ploty**2 + best_right[1]*ploty + best_right[2]
        center_fitx = right_fitx - (450 / 2) # 차선 절반만큼 좌측으로 이동

    # 2. 픽셀 좌표를 Ego 차량 기준의 Local 물리 좌표(Meter)로 변환
    ego_x = ego_transform.location.x
    ego_y = ego_transform.location.y
    ego_yaw = math.radians(ego_transform.rotation.yaw)

    for px_x, px_y in zip(center_fitx, ploty):
        # Local 좌표계: 차의 앞쪽이 X, 오른쪽이 Y
        local_x = (IM_HEIGHT - px_y) * YM_PER_PIX + 1.5 # 1.5는 카메라 장착 위치(후드) 보정값
        local_y = (px_x - (IM_WIDTH / 2)) * XM_PER_PIX
        
        # 3. Local 좌표를 Global 좌표로 변환 (회전 행렬 적용)
        gx = ego_x + local_x * math.cos(ego_yaw) - local_y * math.sin(ego_yaw)
        gy = ego_y + local_x * math.sin(ego_yaw) + local_y * math.cos(ego_yaw)
        
        path_x.append(gx)
        path_y.append(gy)

    # 4. 각 Waypoint에서의 목표 각도(Yaw) 계산
    for i in range(len(path_x) - 1):
        dx = path_x[i+1] - path_x[i]
        dy = path_y[i+1] - path_y[i]
        path_yaw.append(math.atan2(dy, dx))
    path_yaw.append(path_yaw[-1]) # 마지막 점의 각도 복사

    return path_x, path_y, path_yaw


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
        spawn_point = world.get_map().get_spawn_points()[6]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        # [핵심] 오토파일럿(신의 손)을 완전히 꺼버립니다!
        vehicle.set_autopilot(False) 

        # 카메라 세팅
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        camera_bp.set_attribute('fov', str(project_config.CAMERA_FOV))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-10, yaw=0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        # 모듈 인스턴스화
        M, Minv, src_points = get_ipm_matrices(IM_WIDTH, IM_HEIGHT)
        tracker = LaneTracker(window_size=10) # 10프레임 기억력
        controller = VehicleController()      # 스티어링/엑셀 제어기
        
        print("=== Step 08: Full Autonomous Drive (Vision + Control) Started ===")

        while True:
            world.tick()
            image = image_queue.get()
            
            # 1. 인지 (Perception) - 비전 파이프라인
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            frame = array[:, :, :3]

            bev_image = cv2.warpPerspective(frame, M, (IM_WIDTH, IM_HEIGHT), flags=cv2.INTER_LINEAR)
            lane_mask = extract_lane_mask(bev_image)
            polyfit_img, best_left, best_right = fit_polynomial(lane_mask, tracker)

            # 2. 판단 (Planning) - 픽셀을 물리적 궤적으로 변환
            ego_tf = vehicle.get_transform()
            path_x, path_y, path_yaw = pixel_to_global_path(best_left, best_right, ego_tf)

            # 디버깅: 차량 앞 도로에 시스템이 계산한 목표 궤적을 초록색 점으로 그리기
            for gx, gy in zip(path_x, path_y):
                world.debug.draw_point(carla.Location(x=gx, y=gy, z=ego_tf.location.z+0.5), size=0.05, color=carla.Color(0,255,0), life_time=0.1)

            # 3. 제어 (Control) - Stanley Controller에 궤적 전달
            velocity = vehicle.get_velocity()
            current_speed = math.sqrt(velocity.x**2 + velocity.y**2)
            current_pose = (ego_tf.location.x, ego_tf.location.y, math.radians(ego_tf.rotation.yaw))
            
            if len(path_x) > 0:
                control_cmd = controller.run_step(
                    target_speed=8.0, # 8 m/s (약 30km/h) 주행
                    current_speed=current_speed,
                    current_pose=current_pose,
                    path_x=path_x, path_y=path_y, path_yaw=path_yaw
                )
            else:
                # 차선을 완전히 잃어버렸을 경우 (비상 정지)
                control_cmd = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

            vehicle.apply_control(control_cmd)

            # 시각화창 배치
            cv2.imshow("1. Front Camera", frame)
            cv2.imshow("2. Vision Tracking", polyfit_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

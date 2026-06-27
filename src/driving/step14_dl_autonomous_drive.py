"""
File: step14_dl_autonomous_drive.py

Purpose:
    Run CARLA lane-following with the binary U-Net model.

Main Responsibilities:
    - Load best_unet_model.pth.
    - Infer lane masks from the front camera stream.
    - Convert predictions into a BEV planning path and control the vehicle.

Notes:
    Requires CARLA and a trained checkpoint. It is not expected to run when the
    simulator is offline.
"""

import carla
import queue
import numpy as np
import cv2
import math
import torch

from src import config as project_config

# 이전 단계의 핵심 모듈들 임포트
from src.driving.step04_controller import VehicleController
from src.driving.step07_polyfit import get_ipm_matrices, fit_polynomial, LaneTracker, IM_WIDTH, IM_HEIGHT
from src.models.step11_model import UNet

# 픽셀-미터 변환 상수 (step08과 동일)
YM_PER_PIX = 30.0 / 360  
XM_PER_PIX = 3.5 / 450   
CROP_Y = project_config.CROP_Y # U-Net 학습 시 상단 크롭했던 높이

# ==========================================
# 1. 픽셀 -> 글로벌 좌표 변환 함수
# ==========================================
def pixel_to_global_path(best_left, best_right, ego_transform):
    """비전 모듈의 다항식을 차량 제어용 글로벌 (x,y,yaw) 궤적으로 변환"""
    path_x, path_y, path_yaw = [], [], []
    if best_left is None and best_right is None:
        return path_x, path_y, path_yaw

    ploty = np.linspace(IM_HEIGHT - 1, 0, num=20)
    
    if best_left is not None and best_right is not None:
        center_fitx = ((best_left[0]*ploty**2 + best_left[1]*ploty + best_left[2]) + 
                       (best_right[0]*ploty**2 + best_right[1]*ploty + best_right[2])) / 2.0
    elif best_left is not None:
        center_fitx = (best_left[0]*ploty**2 + best_left[1]*ploty + best_left[2]) + (450 / 2)
    else:
        center_fitx = (best_right[0]*ploty**2 + best_right[1]*ploty + best_right[2]) - (450 / 2)

    ego_x, ego_y = ego_transform.location.x, ego_transform.location.y
    ego_yaw = math.radians(ego_transform.rotation.yaw)

    for px_x, px_y in zip(center_fitx, ploty):
        local_x = (IM_HEIGHT - px_y) * YM_PER_PIX + 1.5
        local_y = (px_x - (IM_WIDTH / 2)) * XM_PER_PIX
        gx = ego_x + local_x * math.cos(ego_yaw) - local_y * math.sin(ego_yaw)
        gy = ego_y + local_x * math.sin(ego_yaw) + local_y * math.cos(ego_yaw)
        path_x.append(gx), path_y.append(gy)

    for i in range(len(path_x) - 1):
        path_yaw.append(math.atan2(path_y[i+1] - path_y[i], path_x[i+1] - path_x[i]))
    if path_yaw: path_yaw.append(path_yaw[-1])

    return path_x, path_y, path_yaw

# ==========================================
# 2. 메인 하이브리드 주행 루프
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 U-Net 추론 디바이스: {device}")

    # 1. 딥러닝 모델(U-Net) 로드 및 초기화
    model = UNet(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(project_config.BINARY_UNET_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # 추론 모드

    # 2. CARLA 초기화
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
        spawn_point = world.get_map().get_spawn_points()[12]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(False) # [핵심] 오토파일럿 완전 종료!

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        camera_bp.set_attribute('fov', str(project_config.CAMERA_FOV))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-10, yaw=0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        # 시스템 모듈 인스턴스화
        M, Minv, src_points = get_ipm_matrices(IM_WIDTH, IM_HEIGHT)
        tracker = LaneTracker(window_size=10)
        controller = VehicleController()
        
        print("=== Step 14: Ultimate Hybrid Autonomous Drive Started ===")

        while True:
            world.tick()
            image = image_queue.get()
            
            # ---------------------------------------------------------
            # [Phase 1: Deep Learning Perception]
            # ---------------------------------------------------------
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            frame_bgr = array[:, :, :3]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # U-Net 입력을 위한 전처리 (상단 크롭 및 정규화)
            img_cropped = frame_rgb[CROP_Y:, :, :].astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_cropped).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.sigmoid(output).squeeze().cpu().numpy()
                unet_mask_cropped = np.where(prob >= 0.5, 255, 0).astype(np.uint8)

            # 슬라이딩 윈도우 알고리즘(step07)과 좌표계를 맞추기 위해 잘라냈던 상단을 검은색으로 복원
            unet_mask_full = np.zeros((IM_HEIGHT, IM_WIDTH), dtype=np.uint8)
            unet_mask_full[CROP_Y:, :] = unet_mask_cropped

            # ---------------------------------------------------------
            # [Phase 2: Geometry & Filtering (Crosswalk Slayer)]
            # ---------------------------------------------------------
            # U-Net 마스크를 조감도(BEV)로 변환
            bev_mask = cv2.warpPerspective(unet_mask_full, M, (IM_WIDTH, IM_HEIGHT), flags=cv2.INTER_LINEAR)
            
            # [핵심] 횡단보도/정지선 제거를 위한 수직 모폴로지 연산
            # 아스팔트 노이즈가 없으므로, 가로선만 정확히 부숴버림
            kernel_vertical = np.ones((15, 3), np.uint8)
            cleaned_bev_mask = cv2.morphologyEx(bev_mask, cv2.MORPH_OPEN, kernel_vertical)

            # ---------------------------------------------------------
            # [Phase 3: Planning & Control]
            # ---------------------------------------------------------
            polyfit_img, best_left, best_right = fit_polynomial(cleaned_bev_mask, tracker)
            
            ego_tf = vehicle.get_transform()
            path_x, path_y, path_yaw = pixel_to_global_path(best_left, best_right, ego_tf)

            # 제어 명령 산출
            velocity = vehicle.get_velocity()
            current_speed = math.sqrt(velocity.x**2 + velocity.y**2)
            current_pose = (ego_tf.location.x, ego_tf.location.y, math.radians(ego_tf.rotation.yaw))
            
            if len(path_x) > 0:
                control_cmd = controller.run_step(8.0, current_speed, current_pose, path_x, path_y, path_yaw)
            else:
                control_cmd = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0) # 비상 정지

            vehicle.apply_control(control_cmd)

            # ---------------------------------------------------------
            # 시각화 (디버깅)
            # ---------------------------------------------------------
            cv2.imshow("1. Front Camera", frame_bgr)
            cv2.imshow("2. U-Net Mask (Full)", unet_mask_full)
            cv2.imshow("3. BEV Cleaned Mask", cleaned_bev_mask)
            cv2.imshow("4. Robust Polyfit", polyfit_img)

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

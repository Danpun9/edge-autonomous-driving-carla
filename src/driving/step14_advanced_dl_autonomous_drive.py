"""
File: step14_advanced_dl_autonomous_drive.py

Purpose:
    Run CARLA lane-following with the advanced multi-class segmentation model.

Main Responsibilities:
    - Load an AdvancedUNet or SMPHybridUNet checkpoint.
    - Infer lane class masks from front camera frames.
    - Feed the lane mask through the existing BEV/polyfit/control pipeline.

Notes:
    The selected checkpoint path must match the active model architecture.
    Requires CARLA to be running on localhost:2000.
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
from src.models.step11_advanced_model import AdvancedUNet
from src.models.step22_smp_model import SMPHybridUNet

# 픽셀-미터 변환 상수
YM_PER_PIX = 30.0 / 360  
XM_PER_PIX = 3.5 / 450   
CROP_Y = project_config.CROP_Y

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
# 2. 대망의 메인 하이브리드 주행 루프
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Advanced U-Net 추론 디바이스: {device}")

    # 1. 3채널 다중 클래스 딥러닝 모델 로드
    # model = AdvancedUNet(in_channels=3, out_channels=3).to(device)
    # checkpoint_path = "advanced_best_unet_model.pth"
    # or
    # model = AdvancedUNet(in_channels=3, out_channels=3).to(device)
    # checkpoint_path = "advanced_best_aug_unet_model.pth"
    # or
    model = SMPHybridUNet(encoder_name="resnet50", classes=3).to(device)
    checkpoint_path = project_config.SMP_RESNET50_CHECKPOINT

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() 

    # 2. CARLA 초기화
    client = carla.Client(project_config.CARLA_HOST, project_config.CARLA_PORT)
    client.set_timeout(project_config.CARLA_TIMEOUT_SECONDS)
    client.load_world('Town04')
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
        vehicle.set_autopilot(False) # 오토파일럿 완전 종료

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        camera_bp.set_attribute('fov', str(project_config.CAMERA_FOV))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-10, yaw=0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        M, Minv, src_points = get_ipm_matrices(IM_WIDTH, IM_HEIGHT)
        tracker = LaneTracker(window_size=10)
        controller = VehicleController()
        
        print("=== Step 14: Ultimate Multi-class Hybrid Drive Started ===")

        while True:
            world.tick()
            image = image_queue.get()
            
            # ---------------------------------------------------------
            # [Phase 1: Deep Learning Perception (다중 클래스 추론)]
            # ---------------------------------------------------------
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape((IM_HEIGHT, IM_WIDTH, 4))
            frame_bgr = array[:, :, :3].copy()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            img_cropped = frame_rgb[CROP_Y:, :, :].astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_cropped).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                # [핵심 1] Argmax를 통해 0(배경), 1(차선), 2(횡단보도) 마스크 추출
                pred_mask = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()

            # ---------------------------------------------------------
            # [Phase 2: Semantic Channel Routing (의미론적 라우팅)]
            # ---------------------------------------------------------
            
            # 라우팅 A: 조향을 위한 "차선(Class 1)"만 추출
            lane_mask = np.zeros_like(pred_mask, dtype=np.uint8)
            lane_mask[pred_mask == 1] = 255
            
            # 원상복구 (상단 180px 검은색 패딩) 및 조감도(BEV) 변환
            lane_mask_full = np.zeros((IM_HEIGHT, IM_WIDTH), dtype=np.uint8)
            lane_mask_full[CROP_Y:, :] = lane_mask
            bev_lane_mask = cv2.warpPerspective(lane_mask_full, M, (IM_WIDTH, IM_HEIGHT), flags=cv2.INTER_LINEAR)
            
            # [핵심 2] morphologyEx 삭제! 횡단보도가 이미 걸러졌으므로 순수 차선만 남음.

            # 라우팅 B: 제동을 위한 "횡단보도/정지선(Class 2)" 감지
            # 차량 바로 앞(Bottom Center) ROI(관심 영역) 설정
            crosswalk_roi = pred_mask[130:180, 200:440] 
            crosswalk_pixels = np.sum(crosswalk_roi == 2)
            is_crosswalk_ahead = crosswalk_pixels > 200 # 임계값 픽셀 넘으면 감지

            # ---------------------------------------------------------
            # [Phase 3: Planning & Control (종/횡방향 제어)]
            # ---------------------------------------------------------
            # 1. 횡방향(Steering) 제어: 순수 차선만 들어간 BEV로 궤적 생성
            polyfit_img, best_left, best_right = fit_polynomial(bev_lane_mask, tracker)
            
            ego_tf = vehicle.get_transform()
            path_x, path_y, path_yaw = pixel_to_global_path(best_left, best_right, ego_tf)

            velocity = vehicle.get_velocity()
            current_speed = math.sqrt(velocity.x**2 + velocity.y**2)
            current_pose = (ego_tf.location.x, ego_tf.location.y, math.radians(ego_tf.rotation.yaw))
            
            # Stanley 조향각 계산
            if len(path_x) > 0:
                control_cmd = controller.run_step(8.0, current_speed, current_pose, path_x, path_y, path_yaw)
            else:
                control_cmd = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0) 

            # 2. 종방향(Brake) 제어 개입: 횡단보도 감지 시
            if is_crosswalk_ahead:
                cv2.putText(frame_bgr, "WARNING: CROSSWALK DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                # control_cmd.throttle = 0.0
                # control_cmd.brake = 1.0 # 급제동 (실제로는 부드러운 감속 로직 적용 가능)

            vehicle.apply_control(control_cmd)

            # ---------------------------------------------------------
            # 시각화 (디버깅)
            # ---------------------------------------------------------
            # 시각화용 통합 마스크 만들기 (차선=빨강, 횡단보도=초록)
            color_mask = np.zeros((180, 640, 3), dtype=np.uint8)
            color_mask[pred_mask == 1] = [0, 0, 255] # BGR Red
            color_mask[pred_mask == 2] = [0, 255, 0] # BGR Green
            
            cv2.imshow("1. Front Camera & Semantic Braking", frame_bgr)
            cv2.imshow("2. Semantic Routing (Red:Lane, Green:Stop)", color_mask)
            cv2.imshow("3. BEV Pure Lane Mask", bev_lane_mask)
            cv2.imshow("4. Flawless Polyfit", polyfit_img)

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

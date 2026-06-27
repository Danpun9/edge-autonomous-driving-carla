"""
File: step17_adaptive_cruise_control.py

Purpose:
    Extend the CARLA sensor-fusion demo with adaptive cruise control behavior.

Main Responsibilities:
    - Estimate lead-vehicle distance from YOLO bounding boxes.
    - Adjust target speed based on safe following distance.
    - Override throttle/brake for red lights and emergency braking.

Notes:
    Requires CARLA and trained segmentation/object-detection checkpoints.
"""

import carla
import queue
import numpy as np
import cv2
import math
import torch
from ultralytics import YOLO

from src import config as project_config

# 이전 단계의 모듈들 임포트
from src.driving.step04_controller import VehicleController
from src.driving.step07_polyfit import get_ipm_matrices, fit_polynomial, LaneTracker, IM_WIDTH, IM_HEIGHT
from src.models.step11_advanced_model import AdvancedUNet

YM_PER_PIX = 30.0 / 360  
XM_PER_PIX = 3.5 / 450   
CROP_Y = project_config.CROP_Y
FOV = project_config.CAMERA_FOV

# [ACC 핵심 수학] 핀홀 카메라 초점 거리 계산
# f = (image_width / 2) / tan(FOV / 2)
FOCAL_LENGTH = (IM_WIDTH / 2.0) / math.tan(math.radians(FOV / 2.0))
REAL_CAR_WIDTH = 2.0 # 세상 모든 차의 평균 너비를 2.0m로 가정

# YOLO 클래스 (2: car, 3: motorcycle, 5: bus, 7: truck, 9: traffic light)
VEHICLE_CLASSES = [2, 3, 5, 7]

def pixel_to_global_path(best_left, best_right, ego_transform):
    path_x, path_y, path_yaw = [], [], []
    if best_left is None and best_right is None: return path_x, path_y, path_yaw

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

def classify_traffic_light_color(frame, box):
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return "UNKNOWN"
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))

    r_pix, g_pix = cv2.countNonZero(mask_red), cv2.countNonZero(mask_green)
    if r_pix > g_pix and r_pix > 5: return "RED"
    elif g_pix > r_pix and g_pix > 5: return "GREEN"
    return "UNKNOWN"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 ACC & Sensor Fusion 시스템 부팅 중...")

    unet_model = AdvancedUNet(in_channels=3, out_channels=3).to(device)
    unet_model.load_state_dict(torch.load(project_config.ADVANCED_UNET_CHECKPOINT, map_location=device)['model_state_dict'])
    unet_model.eval()

    yolo_model = YOLO(project_config.YOLO_MODEL_PATH)
    print("✅ 듀얼 코어 AI 로드 완료!")

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
        vehicle.set_autopilot(False)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        camera_bp.set_attribute('fov', str(FOV))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-10, yaw=0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        M, Minv, src_points = get_ipm_matrices(IM_WIDTH, IM_HEIGHT)
        tracker = LaneTracker(window_size=10)
        controller = VehicleController()
        
        print("=== Step 17: Monocular ACC & Fusion Drive Started ===")

        while True:
            world.tick()
            image = image_queue.get()
            
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape((IM_HEIGHT, IM_WIDTH, 4))
            frame_bgr = array[:, :, :3]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            vis_frame = frame_bgr.copy()

            # ---------------------------------------------------------
            # [Phase 1: U-Net 추론]
            # ---------------------------------------------------------
            img_cropped = frame_rgb[CROP_Y:, :, :].astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_cropped).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = unet_model(img_tensor)
                pred_mask = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()

            lane_pixels = np.argwhere(pred_mask == 1)
            lane_center_x = IM_WIDTH // 2 

            if len(lane_pixels) > 0:
                min_y = np.min(lane_pixels[:, 0])
                top_pixels = lane_pixels[lane_pixels[:, 0] < min_y + 15]
                lane_center_x = int(np.mean(top_pixels[:, 1]))

            cv2.line(vis_frame, (lane_center_x, 0), (lane_center_x, IM_HEIGHT), (255, 255, 0), 1, cv2.LINE_AA)

            is_crosswalk_ahead = np.sum(pred_mask[130:180, 200:440] == 2) > 150

            # ---------------------------------------------------------
            # [Phase 2: YOLO 탐지 및 타겟/신호등 매칭]
            # ---------------------------------------------------------
            yolo_results = yolo_model.predict(source=frame_bgr, conf=0.45, classes=VEHICLE_CLASSES + [9], verbose=False)[0]
            
            matched_tl_box = None
            min_tl_dist = float('inf')
            
            lead_vehicle_dist = float('inf')
            lead_vehicle_box = None

            for box in yolo_results.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                box_w = x2 - x1
                box_h = y2 - y1
                center_x = (x1 + x2) // 2
                
                # 가짜 객체 차단
                if box_w < 10 or box_h < 10: continue

                # 1. 신호등 처리
                if cls_id == 9:
                    dist = abs(center_x - lane_center_x)
                    if dist < min_tl_dist and dist < 80:
                        min_tl_dist = dist
                        matched_tl_box = (x1, y1, x2, y2)
                
                # 2. 차량 처리 (ACC Target Selection)
                elif cls_id in VEHICLE_CLASSES:
                    # [핵심] 차선 소실점과의 거리가 가깝고 (내 차선에 있고), 가장 가까운 차량 찾기
                    dist_from_lane = abs(center_x - lane_center_x)
                    if dist_from_lane < 100: # 내 차선 범위 내에 있다면
                        # 핀홀 카메라 모델로 거리(Distance) 추정
                        estimated_dist = (REAL_CAR_WIDTH * FOCAL_LENGTH) / box_w
                        
                        if estimated_dist < lead_vehicle_dist:
                            lead_vehicle_dist = estimated_dist
                            lead_vehicle_box = (x1, y1, x2, y2)

            # 신호등 상태 업데이트
            tl_state = "GREEN" 
            if matched_tl_box:
                tl_state = classify_traffic_light_color(frame_bgr, matched_tl_box)
                x1, y1, x2, y2 = matched_tl_box
                color = (0, 0, 255) if tl_state == "RED" else (0, 255, 0)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_frame, f"EGO_TL: {tl_state}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 앞차 시각화
            if lead_vehicle_box:
                x1, y1, x2, y2 = lead_vehicle_box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(vis_frame, f"TARGET: {lead_vehicle_dist:.1f}m", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # ---------------------------------------------------------
            # [Phase 3: ACC & Sensor Fusion 하이브리드 제어]
            # ---------------------------------------------------------
            lane_mask = np.zeros_like(pred_mask, dtype=np.uint8)
            lane_mask[pred_mask == 1] = 255
            lane_mask_full = np.zeros((IM_HEIGHT, IM_WIDTH), dtype=np.uint8)
            lane_mask_full[CROP_Y:, :] = lane_mask
            bev_lane_mask = cv2.warpPerspective(lane_mask_full, M, (IM_WIDTH, IM_HEIGHT), flags=cv2.INTER_LINEAR)
            polyfit_img, best_left, best_right = fit_polynomial(bev_lane_mask, tracker)
            
            ego_tf = vehicle.get_transform()
            path_x, path_y, path_yaw = pixel_to_global_path(best_left, best_right, ego_tf)

            velocity = vehicle.get_velocity()
            current_speed = math.sqrt(velocity.x**2 + velocity.y**2)
            current_pose = (ego_tf.location.x, ego_tf.location.y, math.radians(ego_tf.rotation.yaw))
            
            # --- 종방향(속도) 목표 설정 ---
            TARGET_CRUISE_SPEED = 8.0 # 기본 크루즈 속도 (m/s)
            dynamic_target_speed = TARGET_CRUISE_SPEED

            if lead_vehicle_dist < float('inf'):
                # 앞차가 있을 때의 ACC 로직
                SAFE_DIST = 10.0 # 유지하고자 하는 안전 거리(m)
                if lead_vehicle_dist < SAFE_DIST:
                    # 너무 가까움 -> 속도를 팍 줄임 (또는 정지)
                    dynamic_target_speed = max(0.0, current_speed - 2.0)
                    cv2.putText(vis_frame, "ACC: BRAKING", (50, IM_HEIGHT - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                elif lead_vehicle_dist < SAFE_DIST + 5.0:
                    # 적당한 거리 -> 속도 유지 (앞차와 동기화)
                    dynamic_target_speed = current_speed
                    cv2.putText(vis_frame, "ACC: FOLLOWING", (50, IM_HEIGHT - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                else:
                    cv2.putText(vis_frame, "ACC: ACCELERATING", (50, IM_HEIGHT - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # 조향 제어 산출 (동적 타겟 속도 반영)
            if len(path_x) > 0:
                control_cmd = controller.run_step(dynamic_target_speed, current_speed, current_pose, path_x, path_y, path_yaw)
            else:
                control_cmd = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0) 

            # --- 최고 우선순위 오버라이드: 신호등 ---
            if tl_state == "RED":
                if is_crosswalk_ahead:
                    cv2.putText(vis_frame, "STOP LINE AHEAD: BRAKE!", (50, IM_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    control_cmd.throttle = 0.0
                    control_cmd.brake = 1.0 
                else:
                    cv2.putText(vis_frame, "RED LIGHT: COASTING...", (50, IM_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    if current_speed > 3.0: control_cmd.throttle = 0.0

            # 초근접 긴급 제동 (AEB)
            if lead_vehicle_dist < 4.0:
                cv2.putText(vis_frame, "EMERGENCY BRAKE!", (IM_WIDTH//2 - 150, IM_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                control_cmd.throttle = 0.0
                control_cmd.brake = 1.0

            vehicle.apply_control(control_cmd)

            # 시각화
            cv2.imshow("1. ACC & Sensor Fusion", vis_frame)
            cv2.imshow("2. Polyfit Planning", polyfit_img)

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

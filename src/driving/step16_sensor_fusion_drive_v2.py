"""
File: step16_sensor_fusion_drive_v2.py

Purpose:
    Run a revised U-Net + YOLO sensor-fusion driving loop in CARLA.

Main Responsibilities:
    - Infer lane/crosswalk classes from the segmentation model.
    - Detect vehicles and traffic lights with YOLO.
    - Apply traffic-light, ACC, and emergency-braking logic.

Notes:
    Requires CARLA, trained model weights, and a compatible YOLO checkpoint.
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

def pixel_to_global_path(best_left, best_right, ego_transform):
    """비전 모듈의 다항식을 차량 제어용 글로벌 (x,y,yaw) 궤적으로 변환"""
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

# ==========================================
# [퓨전 마법] OpenCV 기반 신호등 색상 판별 함수
# ==========================================
def classify_traffic_light_color(frame, box):
    """YOLO 바운딩 박스 내부의 픽셀을 분석하여 적/녹 상태 반환"""
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return "UNKNOWN"
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # 빨간색은 HSV의 양 끝단에 분포함
    mask_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # 초록색
    mask_green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))

    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)

    if red_pixels > green_pixels and red_pixels > 5: return "RED"
    elif green_pixels > red_pixels and green_pixels > 5: return "GREEN"
    return "UNKNOWN"

# ==========================================
# 대망의 메인 하이브리드 주행 루프
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 센서 퓨전 시스템 부팅 중...")

    # 1. 뇌 A (U-Net) 로드
    unet_model = AdvancedUNet(in_channels=3, out_channels=3).to(device)
    unet_model.load_state_dict(torch.load(project_config.ADVANCED_UNET_CHECKPOINT, map_location=device)['model_state_dict'])
    unet_model.eval()

    # 2. 뇌 B (YOLOv8) 로드
    yolo_model = YOLO(project_config.YOLO_MODEL_PATH)
    print("✅ U-Net & YOLO 듀얼 코어 로드 완료!")

    # CARLA 초기화
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
        camera_bp.set_attribute('fov', str(project_config.CAMERA_FOV))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-10, yaw=0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        M, Minv, src_points = get_ipm_matrices(IM_WIDTH, IM_HEIGHT)
        tracker = LaneTracker(window_size=10)
        controller = VehicleController()
        print("=== Step 16: Advanced Sensor Fusion Drive Started ===")

        while True:
            world.tick()
            image = image_queue.get()
            
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape((IM_HEIGHT, IM_WIDTH, 4))
            frame_bgr = array[:, :, :3]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            vis_frame = frame_bgr.copy()

            # ---------------------------------------------------------
            # [Phase 1: U-Net 추론 및 차선 소실점 & 횡단보도 감지]
            # ---------------------------------------------------------
            img_cropped = frame_rgb[CROP_Y:, :, :].astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_cropped).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = unet_model(img_tensor)
                pred_mask = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()

            # 1. 차선 소실점 계산 (Ray Casting용)
            lane_pixels = np.argwhere(pred_mask == 1)
            lane_center_x = IM_WIDTH // 2 

            if len(lane_pixels) > 0:
                min_y = np.min(lane_pixels[:, 0])
                top_pixels = lane_pixels[lane_pixels[:, 0] < min_y + 15]
                lane_center_x = int(np.mean(top_pixels[:, 1]))

            cv2.line(vis_frame, (lane_center_x, 0), (lane_center_x, IM_HEIGHT), (255, 255, 0), 1, cv2.LINE_AA)

            # 2. 전방 횡단보도/정지선 감지 (정지 위치 결정을 위한 U-Net의 조언)
            crosswalk_roi = pred_mask[130:180, 200:440] # 차량 바로 앞 ROI
            is_crosswalk_ahead = np.sum(crosswalk_roi == 2) > 150 # 임계값 설정

            # ---------------------------------------------------------
            # [Phase 2 & 3: YOLO 탐지 및 가짜 객체(False Positive) 필터링]
            # ---------------------------------------------------------
            # conf를 0.3에서 0.45로 올려 쓰레기 데이터 1차 차단
            yolo_results = yolo_model.predict(source=frame_bgr, conf=0.45, classes=[9], verbose=False)[0]
            
            matched_tl_box = None
            min_dist = float('inf')

            for box in yolo_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                box_w = x2 - x1
                box_h = y2 - y1
                
                # [방어 로직 1] 박스가 너무 작으면(멀리 있는 전광판 등) 무시
                if box_w < 8 or box_h < 15:
                    continue
                    
                tl_center_x = (x1 + x2) // 2
                dist = abs(tl_center_x - lane_center_x)
                
                # [방어 로직 2] 차선 수직선과 80픽셀 이내일 때만 내 신호등으로 인정
                if dist < min_dist and dist < 80:
                    min_dist = dist
                    matched_tl_box = (x1, y1, x2, y2)

            tl_state = "GREEN" 
            if matched_tl_box:
                tl_state = classify_traffic_light_color(frame_bgr, matched_tl_box)
                x1, y1, x2, y2 = matched_tl_box
                color = (0, 0, 255) if tl_state == "RED" else (0, 255, 0)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(vis_frame, f"EGO_TL: {tl_state}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # ---------------------------------------------------------
            # [Phase 4: 진정한 하이브리드 제어 (U-Net + YOLO 융합)]
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
            
            # 기본 조향 제어 산출
            if len(path_x) > 0:
                control_cmd = controller.run_step(8.0, current_speed, current_pose, path_x, path_y, path_yaw)
            else:
                control_cmd = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0) 

            # [핵심] 정교한 브레이킹 로직 (Stop Line 기반 정지)
            if tl_state == "RED":
                if is_crosswalk_ahead:
                    # 빨간불이고, 눈앞에 정지선/횡단보도가 보일 때 -> 완벽한 급정거
                    cv2.putText(vis_frame, "STOP LINE AHEAD: BRAKE!", (50, IM_HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    control_cmd.throttle = 0.0
                    control_cmd.brake = 1.0 
                else:
                    # 빨간불이지만 아직 횡단보도가 멀 때 -> 속도를 줄이며 조향 유지 (Coast)
                    cv2.putText(vis_frame, "RED LIGHT: COASTING...", (50, IM_HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    if current_speed > 3.0: # 3km/h 이하로 서행
                        control_cmd.throttle = 0.0
                    else:
                        control_cmd.throttle = 0.2

            vehicle.apply_control(control_cmd)

            # 시각화
            cv2.imshow("1. Sensor Fusion Vision", vis_frame)
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

"""
File: step06_lane_masking.py

Purpose:
    Extract a binary lane mask from a CARLA camera stream using classical
    computer vision.

Main Responsibilities:
    - Transform the camera frame into bird's-eye view.
    - Use HLS color masks, Sobel edges, and morphology to isolate lanes.
    - Display intermediate BEV and lane-mask outputs.

Related Files:
    - step05_ipm.py: provides the IPM concept used here.
    - step07_polyfit.py: reuses and extends the lane-mask pipeline.
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
    src_points = np.float32([
        [40, h],            # 좌측 하단
        [250, h * 0.55],    # 좌측 상단 (소실점 부근)
        [390, h * 0.55],    # 우측 상단
        [600, h]            # 우측 하단
    ])

    dst_points = np.float32([
        [0, h],             # 좌측 하단
        [0, 0],             # 좌측 상단
        [w, 0],             # 우측 상단
        [w, h]              # 우측 하단
    ])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    
    return M, Minv, src_points

def extract_lane_mask(bev_img):
    """
    조감도 이미지에서 색상과 윤곽선을 기준으로 차선만 추출하여 이진화(Binary) 합니다.
    가우시안 블러, 임계치 조절, 모폴로지 연산을 추가하여 노이즈를 제거합니다.
    """
    # 0. 노이즈 제거를 위한 가우시안 블러 적용
    blurred_img = cv2.GaussianBlur(bev_img, (5, 5), 0)

    # 1. HLS 색공간 필터링 (그림자에 강함)
    hls = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1] # Lightness (밝기)
    s_channel = hls[:, :, 2] # Saturation (채도)
    
    # 흰색 차선 추출 (밝기가 매우 높은 픽셀)
    white_mask = cv2.inRange(l_channel, 200, 255)
    
    # 노란색 차선 추출 (CARLA의 중앙선 등) - Hue, Lightness, Saturation 범위 조절
    yellow_mask = cv2.inRange(hls, np.array([10, 0, 100]), np.array([40, 255, 255]))
    
    # 두 색상 마스크 합치기
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # 2. Sobel Edge 필터링 (x방향 수직선 추출)
    gray = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) # x방향 미분
    abs_sobel_x = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))
    
    # 윤곽선 강도가 특정 임계치 이상인 픽셀만 추출 (노이즈 방지를 위해 30->60 상향)
    sobel_mask = np.zeros_like(scaled_sobel)
    sobel_mask[(scaled_sobel >= 60) & (scaled_sobel <= 255)] = 255

    # 3. Color와 Sobel 마스크 결합 (둘 중 하나라도 만족하면 차선으로 인정)
    combined_mask = cv2.bitwise_or(color_mask, sobel_mask)
    
    # 4. 모폴로지 연산 (Morphology)
    kernel = np.ones((3, 3), np.uint8)
    # 침식(Erosion): 자잘한 점 노이즈 제거
    eroded_mask = cv2.erode(combined_mask, kernel, iterations=1)
    # 팽창(Dilation): 남아있는 차선 픽셀들을 다시 굵게 연결
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
    
    return dilated_mask

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
        client.get_trafficmanager(8000).set_synchronous_mode(True)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        camera_bp.set_attribute('fov', str(FOV))
        
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-10, yaw=0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        M, Minv, src_points = get_ipm_matrices(IM_WIDTH, IM_HEIGHT)
        print("=== Step 06: Binary Lane Masking Started ===")

        while True:
            world.tick()
            image = image_queue.get()

            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            frame = array[:, :, :3].copy()

            # 1. BEV 변환
            bev_image = cv2.warpPerspective(frame, M, (IM_WIDTH, IM_HEIGHT), flags=cv2.INTER_LINEAR)

            # 2. 차선 이진화 마스크 추출 (신규 로직)
            lane_mask = extract_lane_mask(bev_image)

            # 3. 시각화
            pts = src_points.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            cv2.imshow("1. Original Camera", frame)
            cv2.imshow("2. BEV Image", bev_image)
            cv2.imshow("3. Lane Binary Mask", lane_mask) # 완전히 까만 배경에 차선만 하얗게 뜨는지 확인

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

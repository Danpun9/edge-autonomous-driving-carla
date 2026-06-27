"""
File: step07_polyfit.py

Purpose:
    Track left/right lane boundaries from a binary BEV lane mask.

Main Responsibilities:
    - Build IPM matrices and extract lane masks.
    - Detect lane pixels with a sliding-window histogram method.
    - Fit polynomial lane curves and smooth them with LaneTracker history.

Related Files:
    - step08_autonomous_drive.py and later driving scripts import these helpers.
"""

import carla
import queue
import numpy as np
import cv2

from src import config as project_config

# ==========================================
# [하이퍼파라미터] 해상도 및 윈도우 설정
# ==========================================
IM_WIDTH = project_config.IMAGE_WIDTH
IM_HEIGHT = project_config.IMAGE_HEIGHT
FOV = project_config.CAMERA_FOV
N_WINDOWS = 9       
MARGIN = 40         
MINPIX = 30         

# (이전 step05, 06의 get_ipm_matrices 함수 유지)
def get_ipm_matrices(w, h):
    src_points = np.float32([[40, h], [250, h * 0.55], [390, h * 0.55], [600, h]])
    dst_points = np.float32([[0, h], [0, 0], [w, 0], [w, h]])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    return M, Minv, src_points

def extract_lane_mask(bev_img):
    blurred = cv2.GaussianBlur(bev_img, (5, 5), 0)
    hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hls[:, :, 1], 200, 255)
    yellow_mask = cv2.inRange(hls, np.array([10, 0, 100]), np.array([40, 255, 255]))
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel_x = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))
    sobel_mask = np.zeros_like(scaled_sobel)
    sobel_mask[(scaled_sobel >= 70) & (scaled_sobel <= 255)] = 255

    combined_mask = cv2.bitwise_or(color_mask, sobel_mask)
    
    # [솔루션 3: 관심 영역(ROI) 마스킹] - 화면 중앙(맨홀 등) 강제 차단
    h, w = combined_mask.shape
    roi_mask = np.ones_like(combined_mask)
    cv2.rectangle(roi_mask, (w//2 - 80, h//2), (w//2 + 80, h), 0, -1) # 중앙 하단 블랙박스 처리
    masked_combined = cv2.bitwise_and(combined_mask, roi_mask)

    kernel_vertical = np.ones((9, 2), np.uint8)
    cleaned_mask = cv2.morphologyEx(masked_combined, cv2.MORPH_OPEN, kernel_vertical)
    kernel_dilate = np.ones((15, 3), np.uint8)
    final_mask = cv2.dilate(cleaned_mask, kernel_dilate, iterations=1)

    return final_mask

# ==========================================
# [핵심] 차선 기억 및 지능형 추정 클래스
# ==========================================
class LaneTracker:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.left_fits = []
        self.right_fits = []
        self.best_left = None
        self.best_right = None
        self.lane_width_px = 450 # BEV 이미지 기준 대략적인 차선 너비 픽셀

    def update(self, left_fit, right_fit, left_pixels, right_pixels):
        # [솔루션 2: 동적 신뢰도 평가] 픽셀 수가 800개 이상이면 신뢰할 수 있는 실선으로 간주
        left_good = left_pixels > 800
        right_good = right_pixels > 800

        # 상식(Sanity Check) 기반 강제 복원
        if left_good and not right_good:
            # 우측 점선이 붕괴됨 -> 좌측 실선을 복사하여 450px 우측으로 평행이동
            right_fit = np.copy(left_fit)
            right_fit[2] += self.lane_width_px
        elif right_good and not left_good:
            # 좌측 점선이 붕괴됨 -> 우측 실선을 복사하여 450px 좌측으로 평행이동
            left_fit = np.copy(right_fit)
            left_fit[2] -= self.lane_width_px
        elif not left_good and not right_good:
            # [솔루션 3: 교차로 증발 대처 (관성 주행)] 양쪽 다 없으면 과거 기억 사용
            left_fit = self.best_left
            right_fit = self.best_right

        # [솔루션 1: 시계열 스무딩] 아웃라이어가 제거된 깨끗한 데이터만 평균 큐에 삽입
        if left_fit is not None:
            self.left_fits.append(left_fit)
            if len(self.left_fits) > self.window_size: self.left_fits.pop(0)
            self.best_left = np.mean(self.left_fits, axis=0)

        if right_fit is not None:
            self.right_fits.append(right_fit)
            if len(self.right_fits) > self.window_size: self.right_fits.pop(0)
            self.best_right = np.mean(self.right_fits, axis=0)

        return self.best_left, self.best_right


def fit_polynomial(binary_warped, tracker):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = int(binary_warped.shape[0]//N_WINDOWS)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    for window in range(N_WINDOWS):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - MARGIN
        win_xleft_high = leftx_current + MARGIN
        win_xright_low = rightx_current - MARGIN
        win_xright_high = rightx_current + MARGIN
        
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2) 
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2) 
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > MINPIX: leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > MINPIX: rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # 현재 프레임의 Raw 피팅
    raw_left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 100 else None
    raw_right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 100 else None

    # [핵심] 트래커를 통한 지능형 보정 수행
    best_left, best_right = tracker.update(raw_left_fit, raw_right_fit, len(leftx), len(rightx))

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

    # 시각화 렌더링 (보정된 최종 곡선 적용)
    if best_left is not None:
        left_fitx = best_left[0]*ploty**2 + best_left[1]*ploty + best_left[2]
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
        cv2.polylines(out_img, pts_left, isClosed=False, color=(0, 255, 255), thickness=4)

    if best_right is not None:
        right_fitx = best_right[0]*ploty**2 + best_right[1]*ploty + best_right[2]
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)
        cv2.polylines(out_img, pts_right, isClosed=False, color=(0, 255, 255), thickness=4)

    return out_img, best_left, best_right

def main():
    # ... (CARLA 초기화 로직 동일) ...
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
        
        # [신규] 트래커 인스턴스 생성 (5프레임 기억)
        tracker = LaneTracker(window_size=10)
        print("=== Step 07: Robust Lane Tracking Started ===")

        while True:
            world.tick()
            image = image_queue.get()

            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            frame = array[:, :, :3]

            bev_image = cv2.warpPerspective(frame, M, (IM_WIDTH, IM_HEIGHT), flags=cv2.INTER_LINEAR)
            lane_mask = extract_lane_mask(bev_image)
            
            # 트래커를 넘겨주어 과거 데이터와 상호작용하도록 함
            polyfit_img, left_fit, right_fit = fit_polynomial(lane_mask, tracker)

            cv2.imshow("1. BEV Image", bev_image)
            cv2.imshow("2. Robust Polyfit", polyfit_img)

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

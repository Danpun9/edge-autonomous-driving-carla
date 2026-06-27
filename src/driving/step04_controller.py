"""
File: step04_controller.py

Purpose:
    Convert a target path and speed into CARLA VehicleControl commands.

Main Responsibilities:
    - Use a PID-style longitudinal controller for throttle and brake.
    - Use a Stanley controller for lateral steering.
    - Return a CARLA control object for the ego vehicle.

Notes:
    This module is imported by multiple driving scripts. Keep the control API
    stable unless all callers are updated together.
"""

import numpy as np
import carla

class VehicleController:
    def __init__(self):
        # ==========================================
        # 1. 횡방향(조향) 제어 파라미터 - Stanley Controller
        # ==========================================
        self.k = 1.5           # Cross-track error 이득 (Gain)
        self.k_soft = 1.0      # 저속에서 분모가 0이 되는 것을 방지하는 소프트닝 상수
        self.max_steer = 1.0   # CARLA 최대 스티어링 값 (정규화: -1.0 ~ 1.0)
        
        # ==========================================
        # 2. 종방향(속도) 제어 파라미터 - PID Controller
        # ==========================================
        self.kp = 0.5          # 비례 제어 (현재 오차 반영)
        self.ki = 0.05         # 적분 제어 (누적 오차 반영, 언덕길 등에서 힘 발휘)
        self.kd = 0.1          # 미분 제어 (오차의 변화량 반영, 급가속/급제동 방지)
        
        self.error_sum = 0.0
        self.prev_error = 0.0

    def run_step(self, target_speed, current_speed, current_pose, path_x, path_y, path_yaw):
        """
        차량의 현재 상태와 목표 궤적을 받아 CARLA의 VehicleControl 객체를 반환합니다.
        - current_pose: (x, y, yaw_rad)
        - path_x, path_y, path_yaw: 플래너가 생성한 궤적의 절대 좌표 배열
        """
        # ------------------------------------------
        # A. 종방향 제어 (PID - Throttle & Brake)
        # ------------------------------------------
        speed_error = target_speed - current_speed
        
        # PID 연산
        p_term = self.kp * speed_error
        self.error_sum += speed_error * 0.05 # dt = 0.05s (20 FPS 기준)
        i_term = self.ki * self.error_sum
        d_term = self.kd * (speed_error - self.prev_error) / 0.05
        self.prev_error = speed_error
        
        acceleration = p_term + i_term + d_term
        
        # 가속도가 양수면 Throttle, 음수면 Brake 적용
        throttle = np.clip(acceleration, 0.0, 1.0)
        brake = np.clip(-acceleration, 0.0, 1.0)

        # ------------------------------------------
        # B. 횡방향 제어 (Stanley Controller - Steer)
        # ------------------------------------------
        # 1. 차량 앞차축(Front Axle)의 위치 계산 (Wheelbase 중심점 보정)
        L = 2.9 # 테슬라 모델 3의 축거(Wheelbase) 약 2.9m
        fx = current_pose[0] + (L / 2.0) * np.cos(current_pose[2])
        fy = current_pose[1] + (L / 2.0) * np.sin(current_pose[2])
        
        # 2. 궤적 상에서 프론트 액슬과 가장 가까운 점(Nearest Point) 찾기
        dx = [fx - icx for icx in path_x]
        dy = [fy - icy for icy in path_y]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)
        
        # 3. Heading Error (각도 오차): 목표 궤적의 방향 - 현재 차량의 방향
        target_yaw = path_yaw[target_idx]
        yaw_diff = target_yaw - current_pose[2]
        # 각도를 -pi ~ pi 사이로 정규화 (한 바퀴 꼬이는 현상 방지)
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff)) 
        
        # 4. Cross-track Error (거리 오차): 궤적과 차량 액슬 사이의 수직 거리
        # 외적(Cross Product)을 사용하여 궤적의 왼쪽/오른쪽 방향성(+) 판단
        front_axle_vec = [-np.cos(current_pose[2] + np.pi/2), -np.sin(current_pose[2] + np.pi/2)]
        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)
        
        # 5. Stanley 제어 법칙 공식
        # 조향각 = 각도 오차 + 아크탄젠트(k * 거리오차 / 현재속도)
        theta_e = yaw_diff
        theta_d = np.arctan2(self.k * error_front_axle, current_speed + self.k_soft)
        
        steer = theta_e + theta_d
        
        # CARLA 조향 명령은 -1.0(좌측 끝) ~ 1.0(우측 끝)으로 제한됨
        steer = np.clip(steer, -self.max_steer, self.max_steer)

        # ------------------------------------------
        # C. 제어 명령(Control Command) 패키징
        # ------------------------------------------
        control = carla.VehicleControl()
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.steer = float(steer)
        control.hand_brake = False
        control.manual_gear_shift = False

        return control

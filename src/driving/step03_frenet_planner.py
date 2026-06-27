"""
File: step03_frenet_planner.py

Purpose:
    Generate and score Frenet-frame candidate paths for lane-following motion
    planning.

Main Responsibilities:
    - Build lateral quintic-polynomial trajectories.
    - Evaluate candidate paths with jerk, time, offset, and speed costs.
    - Reject paths that violate speed, acceleration, lateral, or obstacle
      constraints.

Related Files:
    - step00_master_test.py: integration test entry point.
    - step04_controller.py: consumes selected paths for vehicle control.
"""

import numpy as np

# ==========================================
# [하이퍼파라미터] 비용 함수 가중치 (Weights)
# ==========================================
W_J = 0.1       # 승차감(Jerk) 가중치
W_T = 0.1       # 시간(목표 도달 시간) 가중치
W_D = 1.0       # 차선 중앙 유지(Offset) 가중치
W_V = 1.0       # 목표 속도 유지 가중치

# 차량 물리 한계 (Safety Constraints)
MAX_SPEED = 50.0 / 3.6  # 50 km/h -> m/s
MAX_ACCEL = 2.0         # 최대 가속도 (m/s^2)
MAX_CURVATURE = 1.0     # 최대 조향 곡률

class QuinticPolynomial:
    """승차감 최적화(Jerk 최소화)를 위한 5차 다항식 궤적 생성기"""
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        A = np.array([[time**3, time**4, time**5],
                      [3 * time**2, 4 * time**3, 5 * time**4],
                      [6 * time, 12 * time**2, 20 * time**3]])
        B = np.array([xe - self.a0 - self.a1 * time - self.a2 * time**2,
                      vxe - self.a1 - 2.0 * self.a2 * time,
                      axe - 2.0 * self.a2])
        X = np.linalg.solve(A, B)
        self.a3, self.a4, self.a5 = X[0], X[1], X[2]

    def calc_point(self, t):
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5*t**5
    def calc_first_derivative(self, t):
        return self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3 + 5*self.a5*t**4
    def calc_second_derivative(self, t):
        return 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3
    def calc_third_derivative(self, t):
        return 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2


def check_paths_validity(paths, obstacles):
    """안전 제약 조건 위반 및 장애물 충돌 궤적을 걸러냅니다."""
    valid_paths = []
    OBSTACLE_RADIUS = 2.5  # 장애물 주변 안전 반경 (미터)

    for path in paths:
        # 1. 물리적 한계 검사 (이전과 동일)
        if any(v > MAX_SPEED for v in path['s_d']): continue
        if any(abs(a) > MAX_ACCEL for a in path['s_dd']): continue
        if any(abs(d) > 2.5 for d in path['d']): continue
        
        # 2. 장애물 충돌 검사 (Collision Check) 추가
        collision = False
        for obs in obstacles:
            obs_s, obs_d = obs
            
            # 궤적의 모든 점(0.1초 단위)에 대해 장애물과의 거리 계산
            for i in range(len(path['s'])):
                dist = np.hypot(path['s'][i] - obs_s, path['d'][i] - obs_d)
                if dist < OBSTACLE_RADIUS:
                    collision = True
                    break # 한 번이라도 충돌하면 즉시 검사 중단
            if collision: break
            
        if not collision:
            valid_paths.append(path)

    return valid_paths


def generate_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, target_speed, obstacles=None):
    """장애물 파라미터(obstacles)가 추가된 궤적 생성기"""
    if obstacles is None:
        obstacles = []
        
    frenet_paths = []

    # 1. 후보 샘플링 (d: 횡방향 타겟, Ti: 목표 시간, tv: 목표 속도)
    target_d_list = np.arange(-3.0, 3.0, 1.0) 
    target_t_list = np.arange(4.0, 5.0, 0.2)
    target_v_list = np.arange(target_speed - 2.0, target_speed + 2.0, 1.0)

    for di in target_d_list:
        for Ti in target_t_list:
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            for tv in target_v_list:
                path = {'t': [], 'd': [], 'd_d': [], 'd_dd': [], 'd_ddd': [], 's': [], 's_d': [], 's_dd': [], 's_ddd': [], 'cost': 0.0}
                
                # 웨이포인트 기록
                for t in np.arange(0.0, Ti, 0.1):
                    path['t'].append(t)
                    path['d'].append(lat_qp.calc_point(t))
                    path['d_d'].append(lat_qp.calc_first_derivative(t))
                    path['d_dd'].append(lat_qp.calc_second_derivative(t))
                    path['d_ddd'].append(lat_qp.calc_third_derivative(t)) # 횡방향 Jerk
                    
                    # 단순화된 종방향 모델
                    s_pos = s0 + c_speed * t + 0.5 * ((tv - c_speed)/Ti) * t**2
                    s_vel = c_speed + ((tv - c_speed)/Ti) * t
                    s_acc = (tv - c_speed)/Ti
                    path['s'].append(s_pos)
                    path['s_d'].append(s_vel)
                    path['s_dd'].append(s_acc)
                    path['s_ddd'].append(0.0) # 등가속도이므로 종방향 Jerk는 0

                # ==========================================
                # 2. 비용 함수 (Cost Function) 연산
                # ==========================================
                # 횡방향 Jerk 최소화 (승차감)
                lat_jerk_sq_sum = sum(j**2 for j in path['d_ddd'])
                # 종방향 Jerk 최소화
                lon_jerk_sq_sum = sum(j**2 for j in path['s_ddd'])
                
                lat_cost = (W_J * lat_jerk_sq_sum) + (W_T * Ti) + (W_D * (path['d'][-1]**2))
                lon_cost = (W_J * lon_jerk_sq_sum) + (W_T * Ti) + (W_V * (target_speed - path['s_d'][-1])**2)
                
                path['cost'] = lat_cost + lon_cost
                frenet_paths.append(path)

    # 3. 안전 제약 조건 및 장애물 필터링
    valid_paths = check_paths_validity(frenet_paths, obstacles)
    best_path = min(valid_paths, key=lambda x: x['cost']) if valid_paths else None

    # 디버깅/시각화를 위해 best_path와 모든 valid_paths를 함께 반환
    return best_path, valid_paths

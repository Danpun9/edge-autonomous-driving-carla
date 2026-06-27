"""
File: step00_master_test.py

Purpose:
    Run an early integration test for CARLA-based trajectory planning and
    vehicle control.

Main Responsibilities:
    - Connect to a local CARLA server.
    - Generate candidate Frenet paths with step03_frenet_planner.
    - Apply low-level vehicle control through step04_controller.
    - Visualize planned trajectories in the simulator.

Notes:
    Requires CARLA to be running on localhost:2000. This script is an
    interactive simulation check and is not suitable for headless CI.
"""

import carla
import numpy as np
import time

from src import config as project_config

# 파일명 규칙 반영
from src.driving.step03_frenet_planner import generate_frenet_paths
from src.driving.step04_controller import VehicleController

def draw_trajectories(world, ego_vehicle, frenet_paths):
    """
    [업그레이드 완료] 
    생성된 Frenet 궤적을 차량 기준이 아닌 '실제 도로의 형태(Waypoint)'를 따라 휘어지게 그립니다.
    """
    carla_map = world.get_map()
    ego_loc = ego_vehicle.get_location()
    
    # 1. 현재 차량이 위치한 도로의 중앙선(Waypoint) 획득
    current_wp = carla_map.get_waypoint(ego_loc)

    for i, path in enumerate(frenet_paths[:5]): 
        color = carla.Color(r=0, g=255, b=0) if i % 2 == 0 else carla.Color(r=255, g=0, b=0)
        
        for s, d in zip(path['s'], path['d']):
            # 2. s(종방향 거리)만큼 도로를 따라 전진한 Waypoint 찾기
            if s > 0.1:
                # next(s)는 현재 위치에서 차선을 따라 s 미터 앞의 Waypoint를 리턴합니다.
                next_wps = current_wp.next(s)
                if not next_wps: 
                    continue # 도로가 끝났다면 그리지 않음
                ref_wp = next_wps[0] # 교차로 등에서는 여러 갈래 중 첫 번째(직진) 선택
            else:
                ref_wp = current_wp
                
            # 3. 해당 지점의 절대 좌표(x, y)와 우측 벡터(Right Vector) 가져오기
            ref_loc = ref_wp.transform.location
            right_vec = ref_wp.transform.get_right_vector()
            
            # 4. d(횡방향 거리)만큼 우측 벡터 방향으로 이동하여 최종 글로벌 x, y 계산
            global_x = ref_loc.x + (d * right_vec.x)
            global_y = ref_loc.y + (d * right_vec.y)
            
            # z축(높이)은 살짝 띄워서 도로에 파묻히지 않게 시각화
            point = carla.Location(x=global_x, y=global_y, z=ref_loc.z + 0.5)
            world.debug.draw_point(point, size=0.05, color=color, life_time=0.1)


def main():
    client = carla.Client(project_config.CARLA_HOST, project_config.CARLA_PORT)
    client.set_timeout(project_config.CARLA_TIMEOUT_SECONDS)

    print(client.get_available_maps())

    # client.load_world('Town01')

    world = client.get_world()
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    try:
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_point = world.get_map().get_spawn_points()[0]
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        
        # [수정] 내장 오토파일럿을 꺼버립니다! 이제부터 우리가 직접 조종합니다.
        ego_vehicle.set_autopilot(False) 
        
        # 우리가 만든 제어기 인스턴스화
        controller = VehicleController()
        
        # ==========================================
        # [신규] 전방 40m 지점에 고장 난 트럭(장애물) 강제 스폰
        # ==========================================
        ego_wp = world.get_map().get_waypoint(spawn_point.location)
        obs_wp = ego_wp.next(95.0)[0] # 40미터 전진한 같은 차선의 Waypoint
        
        obs_tf = obs_wp.transform
        obs_tf.location.z += 1.0  # 바닥 반음 방지 (Z축으로 살짝 띄워 스폰)
        
        obs_bp = blueprint_library.find('vehicle.carlamotors.carlacola') # 대형 트럭 소환
        obstacle = world.spawn_actor(obs_bp, obs_tf)
        print("전방 40m 지점에 장애물 트럭 스폰 완료!")

        print("=== Step 00: Master Integration Test Started ===")
        
        while True:
            world.tick() 
            
            velocity = ego_vehicle.get_velocity()
            current_speed = np.sqrt(velocity.x**2 + velocity.y**2)
            ego_tf = ego_vehicle.get_transform()
            ego_loc = ego_tf.location
            
            # ==========================================
            # [핵심] 장애물의 상대 위치(s, d) 계산 (벡터 내적 활용)
            # ==========================================
            obs_loc = obstacle.get_location()
            fwd_vec = ego_tf.get_forward_vector()
            right_vec = ego_tf.get_right_vector()
            
            # 내 차에서 장애물로 향하는 상대 벡터
            rel_vec = np.array([obs_loc.x - ego_loc.x, obs_loc.y - ego_loc.y])
            
            # 내적(Dot Product)을 통해 종방향(s)과 횡방향(d) 투영
            obs_s = np.dot(rel_vec, np.array([fwd_vec.x, fwd_vec.y]))
            obs_d = np.dot(rel_vec, np.array([right_vec.x, right_vec.y]))
            
            # 내 차 앞쪽에 있는 장애물만 플래너에 전달
            obstacles = [(obs_s, obs_d)] if obs_s > 0 else []

            # ==========================================
            # 플래닝 및 제어 (obstacles 파라미터 추가)
            # ==========================================
            best_path, valid_paths = generate_frenet_paths(
                c_speed=current_speed, 
                c_d=0.0, c_d_d=0.0, c_d_dd=0.0, 
                s0=0.0, 
                target_speed=10.0,
                obstacles=obstacles # 인식된 장애물 정보 주입
            )

            if best_path:
                # 시각화 함수에 리스트 형태로 넘겨주기 위해 []로 감쌈
                draw_trajectories(world, ego_vehicle, [best_path])
                
                # [신규 추가] 제어기에 넘겨줄 궤적 포맷 변환
                path_x, path_y, path_yaw = [], [], []
                carla_map = world.get_map()
                ego_loc = ego_vehicle.get_location()
                current_wp = carla_map.get_waypoint(ego_loc)
                
                for s, d in zip(best_path['s'], best_path['d']):
                    if s > 0.1:
                        next_wps = current_wp.next(s)
                        if not next_wps: continue
                        ref_wp = next_wps[0]
                    else:
                        ref_wp = current_wp
                        
                    ref_loc = ref_wp.transform.location
                    right_vec = ref_wp.transform.get_right_vector()
                    
                    global_x = ref_loc.x + (d * right_vec.x)
                    global_y = ref_loc.y + (d * right_vec.y)
                    yaw = np.radians(ref_wp.transform.rotation.yaw)
                    
                    path_x.append(global_x)
                    path_y.append(global_y)
                    path_yaw.append(yaw)

                if path_x:
                    current_pose = (ego_loc.x, ego_loc.y, np.radians(ego_vehicle.get_transform().rotation.yaw))
                    control_cmd = controller.run_step(
                        target_speed=10.0, 
                        current_speed=current_speed, 
                        current_pose=current_pose, 
                        path_x=path_x, 
                        path_y=path_y, 
                        path_yaw=path_yaw
                    )
                    ego_vehicle.apply_control(control_cmd)
            
            draw_trajectories(world, ego_vehicle, valid_paths[:10])

    except KeyboardInterrupt:
        print("\n테스트를 종료합니다.")
    finally:
        # 리소스 정리 시 트럭도 함께 파괴
        if 'obstacle' in locals():
            obstacle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        ego_vehicle.destroy()

if __name__ == '__main__':
    main()

"""
File: check_carla_state.py

Purpose:
    Print basic CARLA world and actor state for quick environment checks.

Main Responsibilities:
    - Connect to a local CARLA server.
    - Read the current map name.
    - Print actor counts and basic world information.

Notes:
    Requires CARLA to be running on localhost:2000.
"""

import carla

from src import config as project_config

def check():
    client = carla.Client(project_config.CARLA_HOST, project_config.CARLA_PORT)
    client.set_timeout(project_config.CARLA_TIMEOUT_SECONDS)
    world = client.get_world()
    spawn_points = world.get_map().get_spawn_points()
    print(f"Total spawn points: {len(spawn_points)}")
    
    actors = world.get_actors()
    vehicles = actors.filter('vehicle.*')
    print(f"Total vehicles in world: {len(vehicles)}")
    
    for i, sp in enumerate(spawn_points[:5]):
        print(f"Spawn point {i}: {sp.location}")

if __name__ == "__main__":
    check()

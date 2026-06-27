"""
File: debug_actors.py

Purpose:
    Debug currently spawned CARLA actors.

Main Responsibilities:
    - Connect to the local CARLA server.
    - Enumerate and print actor information for quick inspection.

Notes:
    Requires CARLA to be running. This script does not create or delete actors.
"""

import carla

from src import config as project_config

client = carla.Client(project_config.CARLA_HOST, project_config.CARLA_PORT)
client.set_timeout(project_config.CARLA_TIMEOUT_SECONDS)
world = client.get_world()

actors = world.get_actors()
vehicles = len(actors.filter('vehicle.*'))
walkers = len(actors.filter('walker.*'))
lights = len(actors.filter('traffic.traffic_light'))

print(f'Vehicles: {vehicles}, Walkers: {walkers}, Lights: {lights}')

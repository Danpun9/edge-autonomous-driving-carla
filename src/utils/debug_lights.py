"""
File: debug_lights.py

Purpose:
    Inspect CARLA traffic-light actors for debugging perception experiments.

Main Responsibilities:
    - Connect to the local CARLA server.
    - Enumerate traffic-light actors.
    - Print state/location information useful for label or perception checks.

Notes:
    Requires CARLA to be running on localhost:2000.
"""

import carla

from src import config as project_config

client = carla.Client(project_config.CARLA_HOST, project_config.CARLA_PORT)
world = client.get_world()

lights = world.get_actors().filter('traffic.traffic_light')
for light in lights:
    boxes = light.get_light_boxes()
    print(f"Light {light.id} has {len(boxes)} light boxes")
    if boxes:
        bb = boxes[0]
        tf = light.get_transform()
        print(f"Transform: {tf}")
        try:
            vertices = bb.get_world_vertices(tf)
            print(f"Vertices: {vertices}")
        except Exception as e:
            print(f"Error: {e}")
    break

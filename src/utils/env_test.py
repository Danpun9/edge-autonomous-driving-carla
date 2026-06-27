"""
File: env_test.py

Purpose:
    Perform a minimal local environment smoke test.

Main Responsibilities:
    - Check that PyTorch can be imported and report CUDA availability.
    - Try connecting to a local CARLA server.

Notes:
    The CARLA check will fail if the simulator is not running.
"""

import torch
import carla

from src import config as project_config

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")


client = carla.Client(project_config.CARLA_HOST, project_config.CARLA_PORT)
client.set_timeout(project_config.CARLA_TIMEOUT_SECONDS)

client_version = client.get_client_version()
server_version = client.get_server_version()

print(f"Client API Version: {client_version}")
print(f"Server Simulator Version: {server_version}")

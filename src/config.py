"""
File: src/config.py

Purpose:
    Centralize project-wide paths and runtime constants used by the CARLA
    Sim2Real autonomous driving scripts.

Main Responsibilities:
    - Define the project root and common dataset/model/output paths.
    - Keep CARLA connection settings in one place.
    - Provide stable defaults that match the original step-based scripts.

Notes:
    For personal overrides, create src/local_config.py with the same constant
    names. That file is ignored by Git.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(*parts: str) -> str:
    """Return an absolute path inside the project root as a string."""
    return str(PROJECT_ROOT.joinpath(*parts))


# CARLA runtime
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_TIMEOUT_SECONDS = 10.0


# Camera and model input defaults
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
PERCEPTION_IMAGE_HEIGHT = 480
CAMERA_FOV = 90
CROP_Y = 180


# Dataset paths
DATASET_BINARY_DIR = project_path("_dataset")
DATASET_MULTICLASS_DIR = project_path("_dataset_multiclass")
DATASET_AUG_UNET_DIR = project_path("_dataset_augmented_unet")
DATASET_YOLO_DIR = project_path("_dataset_yolo")
DATASET_AUG_YOLO_DIR = project_path("_dataset_augmented_yolo")
DATASET_CALIBRATION_DIR = project_path("_dataset_calibration")
DATASET_OTHER_SIM_DIR = project_path("_dataset_other_sim")
DATASET_REAL_DIR = project_path("_dataset_real")
DATASET_REAL_PROCESSED_DIR = project_path("_dataset_real_processing")
DATASET_AIHUB_DIR = project_path("ai_hub_dataset")
DATASET_AIHUB_EVAL_DIR = project_path("_dataset_aihub_eval")
OUT_PERCEPTION_DATA_DIR = project_path("_out_perception_data")


# Output artifact directories
BENCHMARK_OUTPUT_DIR = project_path("benchmark_results")
ONNX_OUTPUT_DIR = project_path("onnx_models")
ONNX_QUANTIZED_DIR = project_path("onnx_quantized")
TENSORRT_ENGINE_DIR = project_path("tensorrt_engines")


# Model and engine paths
YOLO_MODEL_PATH = project_path("yolov8n.pt")
YOLO_EXPERIMENT_CHECKPOINT = project_path("runs", "detect", "train2", "weights", "best.pt")
BINARY_UNET_CHECKPOINT = project_path("best_unet_model.pth")
ADVANCED_UNET_CHECKPOINT = project_path("advanced_best_unet_model.pth")
ADVANCED_AUG_UNET_CHECKPOINT = project_path("advanced_best_aug_unet_model.pth")
SMP_RESNET34_CHECKPOINT = project_path("smp_res34_best_aug_unet_model.pth")
SMP_RESNET50_CHECKPOINT = project_path("smp_best_aug_unet_model.pth")
UFLD_CULANE_CHECKPOINT = project_path("culane_res34.pth")
TENSORRT_FP16_ENGINE = project_path("tensorrt_engines", "ResNet34_Aug_FP16.engine")
TENSORRT_INT8_ENGINE = project_path("tensorrt_engines", "ResNet34_Aug_INT8.engine")
RESNET34_ONNX_MODEL = project_path("onnx_models", "ResNet34_Aug.onnx")
OTHER_SIM_DRIVE_TEST_VIDEO = project_path("_dataset_other_sim", "drive_test2.mp4")
RESULT_DRIVE_TEST_VIDEO = project_path("result_drive_test2.mp4")


# Training defaults
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4


try:
    from .local_config import *  # noqa: F401,F403
except ImportError:
    pass

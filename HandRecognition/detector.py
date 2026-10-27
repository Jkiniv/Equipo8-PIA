import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def initialize_detectors(pose_model_path, hand_model_path, options_config):
  """Verifica la existencia de los modelos y crea los detectores de MediaPipe."""
  if not os.path.isfile(pose_model_path):
    raise FileNotFoundError(
        "No se encontró el modelo de Pose Landmarker:\n" f"{pose_model_path}"
    )

  if not os.path.isfile(hand_model_path):
    raise FileNotFoundError(
        "No se encontró el modelo de Hand Landmarker:\n" f"{hand_model_path}"
    )

  pose_base_options = python.BaseOptions(
      model_asset_path=str(pose_model_path)
  )
  pose_options = vision.PoseLandmarkerOptions(
      base_options=pose_base_options,
      running_mode=vision.RunningMode.VIDEO,
      num_poses=1,
      min_pose_detection_confidence=options_config[
          "POSE_DETECTION_CONFIDENCE"
      ],
      min_pose_presence_confidence=options_config["POSE_PRESENCE_CONFIDENCE"],
      min_tracking_confidence=options_config["POSE_TRACKING_CONFIDENCE"],
      output_segmentation_masks=False,
  )

  hand_base_options = python.BaseOptions(
      model_asset_path=str(hand_model_path)
  )
  hand_options = vision.HandLandmarkerOptions(
      base_options=hand_base_options,
      running_mode=vision.RunningMode.VIDEO,
      num_hands=2,
      min_hand_detection_confidence=options_config[
          "HAND_DETECTION_CONFIDENCE"
      ],
      min_hand_presence_confidence=options_config["HAND_PRESENCE_CONFIDENCE"],
      min_tracking_confidence=options_config["HAND_TRACKING_CONFIDENCE"],
  )

  pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
  hand_detector = vision.HandLandmarker.create_from_options(hand_options)

  return pose_detector, hand_detector
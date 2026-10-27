import cv2
import numpy as np
from utils import (
    HAND_CONNECTIONS,
    draw_line,
    draw_point,
    landmark_is_visible,
    normalized_landmark_to_array,
    select_hand_near_pose_wrist,
    smooth_hand,
    smooth_point,
)


class ArmHandProcessor:

  def __init__(
      self, pose_detector, hand_detector, target_arm, config, use_smoothing=True
  ):
    self.pose_detector = pose_detector
    self.hand_detector = hand_detector
    self.target_arm = target_arm.lower()
    self.use_smoothing = use_smoothing

    self.smoothing_alpha = config.get("SMOOTHING_ALPHA", 0.35)
    self.max_hand_distance = config.get("MAX_HAND_WRIST_DISTANCE", 0.30)

    # Configurar índices según el brazo seleccionado
    if self.target_arm == "left":
      self.shoulder_idx, self.elbow_idx, self.wrist_idx = 11, 13, 15
    elif self.target_arm == "right":
      self.shoulder_idx, self.elbow_idx, self.wrist_idx = 12, 14, 16
    else:
      raise ValueError('TARGET_ARM debe ser "left" o "right".')

    self.hand_wrist_idx = 0

    # Estados anteriores para suavizado
    self.prev_shoulder = None
    self.prev_elbow = None
    self.prev_wrist = None
    self.prev_hand = None

  def reset_smoothing(self):
    self.prev_shoulder = self.prev_elbow = self.prev_wrist = self.prev_hand = (
        None
    )

  def process_frame(self, mp_image, frame, timestamp_ms):
    height, width, _ = frame.shape

    # Ejecutar detectores de MediaPipe
    pose_result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
    hand_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)

    arm_detected = False
    hand_detected = False
    shoulder_coords = elbow_coords = wrist_coords = hand_coords = None
    hand_label = "Unknown"
    distance = None

    # 1. Procesar Pose (Brazo)
    if pose_result.pose_landmarks:
      pose_lm = pose_result.pose_landmarks[0]
      s_land = pose_lm[self.shoulder_idx]
      e_land = pose_lm[self.elbow_idx]
      w_land = pose_lm[self.wrist_idx]

      if (
          landmark_is_visible(s_land)
          and landmark_is_visible(e_land)
          and landmark_is_visible(w_land)
      ):
        arm_detected = True
        shoulder_coords = normalized_landmark_to_array(s_land)
        elbow_coords = normalized_landmark_to_array(e_land)
        wrist_coords = normalized_landmark_to_array(w_land)

        # Seleccionar mano correspondiente
        sel_hand, hand_label, distance = select_hand_near_pose_wrist(
            hand_result, w_land, self.max_hand_distance, self.hand_wrist_idx
        )

        if sel_hand is not None:
          hand_detected = True
          hand_coords = np.array(
              [normalized_landmark_to_array(lm) for lm in sel_hand],
              dtype=np.float32,
          )

    # 2. Aplicar Suavizado
    if arm_detected:
      if self.use_smoothing:
        shoulder_coords = smooth_point(
            self.prev_shoulder, shoulder_coords, self.smoothing_alpha
        )
        elbow_coords = smooth_point(
            self.prev_elbow, elbow_coords, self.smoothing_alpha
        )
        wrist_coords = smooth_point(
            self.prev_wrist, wrist_coords, self.smoothing_alpha
        )
      self.prev_shoulder, self.prev_elbow, self.prev_wrist = (
          shoulder_coords.copy(),
          elbow_coords.copy(),
          wrist_coords.copy(),
      )
    else:
      self.prev_shoulder = self.prev_elbow = self.prev_wrist = None

    if hand_detected:
      if self.use_smoothing:
        hand_coords = smooth_hand(
            self.prev_hand, hand_coords, self.smoothing_alpha
        )
      self.prev_hand = hand_coords.copy()
    else:
      self.prev_hand = None

    # 3. Dibujar sobre el frame
    if arm_detected:
      color_arm = (0, 255, 0)
      draw_line(
          frame,
          shoulder_coords,
          elbow_coords,
          width,
          height,
          color_arm,
          thickness=4,
      )
      draw_line(
          frame,
          elbow_coords,
          wrist_coords,
          width,
          height,
          color_arm,
          thickness=4,
      )
      draw_point(frame, shoulder_coords, width, height, (255, 0, 0), 8)
      draw_point(frame, elbow_coords, width, height, (0, 165, 255), 8)
      draw_point(frame, wrist_coords, width, height, (0, 0, 255), 8)

    if hand_detected:
      for start_idx, end_idx in HAND_CONNECTIONS:
        draw_line(
            frame,
            hand_coords[start_idx],
            hand_coords[end_idx],
            width,
            height,
            (255, 0, 255),
            thickness=2,
        )
      for idx, lm in enumerate(hand_coords):
        pt_color = (0, 0, 255) if idx == self.hand_wrist_idx else (0, 255, 255)
        draw_point(frame, lm, width, height, pt_color, radius=4)

    return {
        "arm_detected": arm_detected,
        "hand_detected": hand_detected,
        "shoulder": shoulder_coords,
        "elbow": elbow_coords,
        "wrist": wrist_coords,
        "hand": hand_coords,
        "hand_label": hand_label,
        "distance": distance,
    }
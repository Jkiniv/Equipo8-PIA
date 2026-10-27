from pathlib import Path
import time
import cv2
import mediapipe as mp
import numpy as np

from detector import initialize_detectors
from processor import ArmHandProcessor
from utils import print_robot_coordinates

# ============================================================
# CONFIGURACIÓN
# ============================================================
POSE_MODEL = Path(__file__).parent / "modelos" / "pose_landmarker_full.task"
HAND_MODEL = Path(__file__).parent / "modelos" / "hand_landmarker.task"

CAMERA_INDEX = 0  # O tu URL IP / índice V4L2
TARGET_ARM = "right"

CONFIG = {
    "POSE_DETECTION_CONFIDENCE": 0.2,
    "POSE_PRESENCE_CONFIDENCE": 0.2,
    "POSE_TRACKING_CONFIDENCE": 0.2,
    "HAND_DETECTION_CONFIDENCE": 0.2,
    "HAND_PRESENCE_CONFIDENCE": 0.2,
    "HAND_TRACKING_CONFIDENCE": 0.2,
    "SMOOTHING_ALPHA": 0.35,
    "MAX_HAND_WRIST_DISTANCE": 0.30,
}

PRINT_INTERVAL_SECONDS = 0.10

# ============================================================
# INICIO DE CÁMARA Y DETECTORES
# ============================================================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
  cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
  raise RuntimeError("No se pudo abrir la cámara.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

pose_detector, hand_detector = initialize_detectors(
    POSE_MODEL, HAND_MODEL, CONFIG
)
processor = ArmHandProcessor(
    pose_detector, hand_detector, TARGET_ARM, CONFIG, use_smoothing=True
)

# Variables de control temporal
program_start_time = time.perf_counter()
last_timestamp_ms = -1
last_print_time = 0.0
fps_counter, fps_display, fps_start_time = 0, 0.0, time.perf_counter()

print("=" * 70)
print(f"Sistema iniciado | Brazo objetivo: {TARGET_ARM.upper()}")
print("=" * 70)

# ============================================================
# BUCLE PRINCIPAL
# ============================================================
try:
  while True:
    success, frame = cap.read()
    if not success:
      print("ERROR: Frame vacío de la cámara.")
      break
    # Downscale processing resolution to 640x360 for faster detection


    processing_width, processing_height = 640, 360
    small_frame = cv2.resize(frame, (processing_width, processing_height))

# Run MediaPipe on the smaller frame
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

# (Optional) Scale the resulting normalized coordinates back up if drawing on the 1280x720 frame,
# or simply keep everything at 640x360.
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Cálculo de FPS
    fps_counter += 1
    now = time.perf_counter()
    if (now - fps_start_time) >= 1.0:
      fps_display = fps_counter / (now - fps_start_time)
      fps_counter = 0
      fps_start_time = now

    # Timestamp de MediaPipe
    timestamp_ms = int((now - program_start_time) * 1000)
    if timestamp_ms <= last_timestamp_ms:
      timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = timestamp_ms

    # Preparar imagen para MediaPipe
    rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Procesar lógica mediante el procesador modular
    data = processor.process_frame(mp_image, frame, timestamp_ms)

    # Textos de interfaz generales
    cv2.putText(
        frame,
        f"Brazo: {TARGET_ARM.upper()}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"FPS: {fps_display:.1f}",
        (width - 150, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
    )

    status_arm_text = (
        "BRAZO DETECTADO" if data["arm_detected"] else "BRAZO NO DETECTADO"
    )
    color_arm = (0, 255, 0) if data["arm_detected"] else (0, 0, 255)
    cv2.putText(
        frame,
        status_arm_text,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        color_arm,
        2,
    )

    # Imprimir consola de forma espaciada
    if (
        data["arm_detected"]
        and data["hand_detected"]
        and (now - last_print_time >= PRINT_INTERVAL_SECONDS)
    ):
      print_robot_coordinates(
          timestamp_ms=timestamp_ms,
          target_arm=TARGET_ARM,
          shoulder=data["shoulder"],
          elbow=data["elbow"],
          wrist=data["wrist"],
          hand=data["hand"],
      )
      last_print_time = now

    cv2.imshow("MediaPipe - Control de robot", frame)
    if (cv2.waitKey(1) & 0xFF) == 27:
      print("\nPrograma detenido por el usuario.")
      break

finally:
  cap.release()
  pose_detector.close()
  hand_detector.close()
  cv2.destroyAllWindows()
  print("Recursos liberados correctamente.")
# Simulación en tiempo real de URDF controlado por MediaPipe
from pathlib import Path
import time
import cv2
import mediapipe as mp
import numpy as np
import roboticstoolbox as rtb   
import swift

from detector import initialize_detectors
from processor import ArmHandProcessor
from utils import print_robot_coordinates

# ============================================================
# CONFIGURACIÓN Y RUTAS
# ============================================================
URDF_FILENAME = "hachi.urdf"  # O "humanoid_shoulder_arm_l_shape.urdf"
urdf_path = Path(__file__).parent / URDF_FILENAME

POSE_MODEL = Path(__file__).parent / "modelos" / "pose_landmarker_full.task"
HAND_MODEL = Path(__file__).parent / "modelos" / "hand_landmarker.task"

CAMERA_INDEX = 0
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
# INICIALIZACIÓN DEL ROBOT Y ENTORNO SWIFT
# ============================================================
robot = rtb.ERobot.URDF(str(urdf_path))
print(robot)

env = swift.Swift()
env.launch(realtime=True)
env.add(robot)

# ============================================================
# INICIO DE CÁMARA Y DETECTORES MEDIAPIPE
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

program_start_time = time.perf_counter()
last_timestamp_ms = -1
last_print_time = 0.0
fps_counter, fps_display, fps_start_time = 0, 0.0, time.perf_counter()

print("=" * 70)
print(f"Sistema iniciado | Control en tiempo real de {TARGET_ARM.upper()}")
print("Presiona 'ESC' en la ventana de video para salir.")
print("=" * 70)

# ============================================================
# BUCLE PRINCIPAL UNIFICADO
# ============================================================
try:
  while True:
    success, frame = cap.read()
    if not success:
      print("ERROR: Frame vacío de la cámara.")
      break

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

    # Preparar imagen
    rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Procesar frames mediante el procesador modular
    data = processor.process_frame(mp_image, frame, timestamp_ms)

    # --- CINEMÁTICA / MAPEO DE COORDENADAS A ARTICULACIONES ---
    q_target = np.zeros(robot.n)

    if data["arm_detected"] and "shoulder" in data and "elbow" in data and "wrist" in data:
      shoulder = np.array(data["shoulder"])
      elbow = np.array(data["elbow"])
      wrist = np.array(data["wrist"])

      # Vectores corporales normalizados devueltos por el procesador
      v_upper = elbow - shoulder
      v_forearm = wrist - elbow

      # Mapeo geométrico básico hacia las 6 articulaciones del robot:
      if robot.n >= 1:
        q_target[0] = np.arctan2(v_upper[0], -v_upper[2])  # Joint 1: Base Yaw
      if robot.n >= 2:
        q_target[1] = np.arctan2(v_upper[1], np.linalg.norm([v_upper[0], v_upper[2]]))  # Joint 2: Shoulder Pitch
      if robot.n >= 3:
        cos_angle = np.dot(v_upper, v_forearm) / (np.linalg.norm(v_upper) * np.linalg.norm(v_forearm) + 1e-6)
        q_target[2] = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Joint 3: Elbow Pitch

      # Si tienes datos de la mano para el efector final (Joint 6 / pinza)
      if data["hand_detected"] and "hand" in data and robot.n >= 6:
        # Ejemplo: usar distancia de apertura o posición relativa de la mano para abrir/cerrar pinza
        q_target[5] = 0.5  # Valor de prueba o mapeado según la pinza

    # Respetar restricciones físicas del URDF
    q_target = np.clip(q_target, robot.qlim[0, :], robot.qlim[1, :])

    # Actualizar estado del robot y refrescar la ventana de Swift
    robot.q = q_target
    env.step(0.01)

    # Textos de interfaz en OpenCV
    cv2.putText(frame, f"Brazo: {TARGET_ARM.upper()}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps_display:.1f}", (width - 150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
    
    status_arm_text = "BRAZO DETECTADO" if data["arm_detected"] else "BRAZO NO DETECTADO"
    color_arm = (0, 255, 0) if data["arm_detected"] else (0, 0, 255)
    cv2.putText(frame, status_arm_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color_arm, 2)

    # Consola periódica
    if data["arm_detected"] and data["hand_detected"] and (now - last_print_time >= PRINT_INTERVAL_SECONDS):
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
  env.hold()
  print("Recursos liberados correctamente.")
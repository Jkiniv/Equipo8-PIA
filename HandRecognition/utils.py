import cv2
import numpy as np


def normalized_landmark_to_array(landmark):
  """Convierte un landmark de MediaPipe en un arreglo [x, y, z]."""
  return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)


def normalized_to_pixel(point, width, height):
  """Convierte las coordenadas normalizadas X/Y a píxeles."""
  x_pixel = int(point[0] * width)
  y_pixel = int(point[1] * height)
  return x_pixel, y_pixel


def landmark_is_visible(landmark, minimum_visibility=0.2):
  """Comprueba visibility y presence del landmark de Pose."""
  visibility = getattr(landmark, "visibility", 1.0)
  presence = getattr(landmark, "presence", 1.0)
  return visibility >= minimum_visibility and presence >= minimum_visibility


def calculate_wrist_distance(hand_landmarks, pose_wrist, hand_wrist_index=0):
  """Calcula la distancia 2D normalizada entre la mano y la muñeca de pose."""
  hand_wrist = hand_landmarks[hand_wrist_index]
  dx = hand_wrist.x - pose_wrist.x
  dy = hand_wrist.y - pose_wrist.y
  distance = np.sqrt(dx**2 + dy**2)
  return float(distance)


def select_hand_near_pose_wrist(
    hand_result, pose_wrist, maximum_distance, hand_wrist_index=0
):
  """Selecciona la mano más cercana a la muñeca del brazo elegido."""
  if not hand_result.hand_landmarks:
    return None, None, None

  nearest_hand = None
  nearest_hand_index = None
  nearest_distance = float("inf")

  for hand_index, hand_landmarks in enumerate(hand_result.hand_landmarks):
    distance = calculate_wrist_distance(
        hand_landmarks, pose_wrist, hand_wrist_index
    )
    if distance < nearest_distance:
      nearest_distance = distance
      nearest_hand = hand_landmarks
      nearest_hand_index = hand_index

  if nearest_distance > maximum_distance:
    return None, None, nearest_distance

  handedness_label = "Unknown"
  if (
      hand_result.handedness
      and nearest_hand_index is not None
      and nearest_hand_index < len(hand_result.handedness)
      and hand_result.handedness[nearest_hand_index]
  ):
    handedness_label = (
        hand_result.handedness[nearest_hand_index][0].category_name
    )

  return nearest_hand, handedness_label, nearest_distance


def smooth_point(previous_point, current_point, alpha):
  """Suavizado exponencial de un punto 3D."""
  if previous_point is None:
    return current_point.copy()
  return alpha * current_point + (1.0 - alpha) * previous_point


def smooth_hand(previous_hand, current_hand, alpha):
  """Aplica suavizado exponencial a los 21 landmarks de la mano."""
  if previous_hand is None:
    return current_hand.copy()
  return alpha * current_hand + (1.0 - alpha) * previous_hand


def draw_point(image, point, width, height, color, radius=6):
  """Dibuja un punto normalizado sobre la imagen."""
  x_pixel, y_pixel = normalized_to_pixel(point, width, height)
  cv2.circle(image, (x_pixel, y_pixel), radius, color, -1)


def draw_line(image, point_a, point_b, width, height, color, thickness=3):
  """Dibuja una línea entre dos puntos normalizados."""
  pixel_a = normalized_to_pixel(point_a, width, height)
  pixel_b = normalized_to_pixel(point_b, width, height)
  cv2.line(image, pixel_a, pixel_b, color, thickness)


def format_vector(vector):
  """Formatea un vector [x, y, z] para mostrarlo en terminal."""
  return f"[{float(vector[0]):.5f}, {float(vector[1]):.5f}, {float(vector[2]):.5f}]"


def print_robot_coordinates(
    timestamp_ms, target_arm, shoulder, elbow, wrist, hand
):
  """Imprime las coordenadas detectadas en consola."""
  print("\n" + "=" * 70)
  print(f"Timestamp: {timestamp_ms} ms")
  print(f"Brazo seleccionado: {target_arm.upper()}")
  print(f"shoulder = {format_vector(shoulder)}")
  print(f"elbow    = {format_vector(elbow)}")
  print(f"wrist    = {format_vector(wrist)}")
  print("hand = [")
  for index, landmark in enumerate(hand):
    print(f"    {index:02d}: {format_vector(landmark)}")
  print("]")


HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Pulgar
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Índice
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Medio
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Anular
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Meñique
    (0, 17),  # Base palma
]
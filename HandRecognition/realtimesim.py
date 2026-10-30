"""Control en tiempo real de un robot URDF de 7 joints con MediaPipe.

Controles en la ventana OpenCV:
- C: calibrar brazo neutral, abajo y recto
- H: calibrar brazo horizontal y recto
- E: calibrar brazo horizontal con codo cercano a 90 grados
- X: reiniciar calibracion del brazo
- F: calibrar mano cerrada
- O: calibrar mano abierta
- R: reiniciar calibracion de la garra
- ESC: salir
"""

from pathlib import Path
import platform
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
# RUTAS Y CONFIGURACION
# ============================================================

BASE_DIRECTORY = Path(__file__).parent
URDF_PATH = BASE_DIRECTORY / "hachi.urdf"
POSE_MODEL = BASE_DIRECTORY / "modelos" / "pose_landmarker_full.task"
HAND_MODEL = BASE_DIRECTORY / "modelos" / "hand_landmarker.task"

CAMERA_INDEX = 0
TARGET_ARM = "right"

CONFIG = {
    # MediaPipe Pose
    "POSE_DETECTION_CONFIDENCE": 0.4,
    "POSE_PRESENCE_CONFIDENCE": 0.4,
    "POSE_TRACKING_CONFIDENCE": 0.4,

    # MediaPipe Hand
    "HAND_DETECTION_CONFIDENCE": 0.2,
    "HAND_PRESENCE_CONFIDENCE": 0.2,
    "HAND_TRACKING_CONFIDENCE": 0.2,
    "NUM_HANDS": 2,

    # Camara e inferencia
    "CAMERA_WIDTH": 1280,
    "CAMERA_HEIGHT": 720,
    "CAMERA_FPS": 30,
    "PROCESSING_WIDTH": 640,
    "PROCESSING_HEIGHT": 360,

    # Procesamiento y tolerancias
    "SMOOTHING_ALPHA": 0.35,
    "MAX_HAND_WRIST_DISTANCE": 0.30,
    "LANDMARK_VISIBILITY_THRESHOLD": 0.30,
    "ARM_LOST_FRAME_TOLERANCE": 5,
    "HAND_LOST_FRAME_TOLERANCE": 5,
    "TORSO_LOST_FRAME_TOLERANCE": 5,
    "LOST_FRAME_TOLERANCE": 5,

    # Dibujo
    "DRAW_ARM": True,
    "DRAW_HAND": True,
    "DRAW_FULL_HAND": False,
    "DRAW_TORSO": True,
    "DRAW_TORSO_AXES": True,
    "TORSO_AXIS_LENGTH": 0.12,

    # Calibracion de brazo
    "ARM_CALIBRATION_SAMPLES": 30,
    "MIN_SHOULDER_CALIBRATION_RANGE_DEG": 35.0,
    "MIN_ELBOW_CALIBRATION_RANGE_DEG": 35.0,
    "ARM_CONTROL_SMOOTHING_ALPHA": 0.25,
    "ARM_DEAD_ZONE_DEG": 3.0,
    "SHOULDER_MAX_DEG": 90.0,
    "ELBOW_MAX_DEG": 90.0,

    # Evita que el plano del hombro cambie cuando el brazo esta abajo,
    # donde el azimut es geometricamente inestable.
    "SHOULDER_PLANE_UPDATE_MIN_DEG": 10.0,

    # Orientacion del plano del hombro respecto al URDF.
    "SHOULDER_PLANE_SIGN": 1.0,

    # Girar el plano 180 grados para apuntar al lado contrario.
    "SHOULDER_PLANE_OFFSET_DEG": 180.0,

    "SHOULDER_ELEVATION_SIGN": 1.0,
    "ELBOW_SIGN": 1.0,

    # Calibracion de garra
    "GRIPPER_CALIBRATION_SAMPLES": 30,
    "GRIPPER_MIN_CALIBRATION_RANGE": 0.20,
    "GRIPPER_SMOOTHING_ALPHA": 0.30,
    "GRIPPER_REVERSED": False,

    # Diagnostico y Swift
    "DEBUG_PRINT_COORDINATES": False,
    "PRINT_INTERVAL_SECONDS": 1.0,
    "SHOW_PROCESSING_TIME": True,
    "SWIFT_STEP_SECONDS": 0.01,
}

REQUIRED_CONFIG_KEYS = {
    "POSE_DETECTION_CONFIDENCE",
    "POSE_PRESENCE_CONFIDENCE",
    "POSE_TRACKING_CONFIDENCE",
    "HAND_DETECTION_CONFIDENCE",
    "HAND_PRESENCE_CONFIDENCE",
    "HAND_TRACKING_CONFIDENCE",
    "CAMERA_WIDTH",
    "CAMERA_HEIGHT",
    "CAMERA_FPS",
    "PROCESSING_WIDTH",
    "PROCESSING_HEIGHT",
    "ARM_CALIBRATION_SAMPLES",
    "MIN_SHOULDER_CALIBRATION_RANGE_DEG",
    "MIN_ELBOW_CALIBRATION_RANGE_DEG",
    "ARM_CONTROL_SMOOTHING_ALPHA",
    "ARM_DEAD_ZONE_DEG",
    "SHOULDER_MAX_DEG",
    "ELBOW_MAX_DEG",
    "SHOULDER_PLANE_UPDATE_MIN_DEG",
    "SHOULDER_PLANE_SIGN",
    "SHOULDER_ELEVATION_SIGN",
    "ELBOW_SIGN",
    "GRIPPER_CALIBRATION_SAMPLES",
    "GRIPPER_MIN_CALIBRATION_RANGE",
    "GRIPPER_SMOOTHING_ALPHA",
    "GRIPPER_REVERSED",
    "DEBUG_PRINT_COORDINATES",
    "PRINT_INTERVAL_SECONDS",
    "SHOW_PROCESSING_TIME",
    "SWIFT_STEP_SECONDS",
}


def validate_config(config, required_keys, config_name):
    missing_keys = sorted(
        key
        for key in required_keys
        if key not in config
    )

    if missing_keys:
        raise KeyError(
            f"Faltan claves en {config_name}: "
            + ", ".join(missing_keys)
        )


validate_config(CONFIG, REQUIRED_CONFIG_KEYS, "realtimesim.py")


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def open_camera(camera_index):
    system_name = platform.system().lower()

    if system_name == "windows":
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    elif system_name == "linux":
        capture = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    else:
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        raise RuntimeError("No se pudo abrir la camara.")

    return capture


def safe_angle_between(vector_a, vector_b):
    vector_a = np.asarray(vector_a, dtype=np.float64)
    vector_b = np.asarray(vector_b, dtype=np.float64)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a <= 1e-6 or norm_b <= 1e-6:
        return None

    cosine = np.dot(vector_a, vector_b) / (norm_a * norm_b)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def smooth_scalar(previous_value, current_value, alpha):
    if current_value is None:
        return previous_value
    if previous_value is None:
        return float(current_value)
    return float(alpha * current_value + (1.0 - alpha) * previous_value)


def normalize_gripper_openness(raw_value, closed_value, open_value):
    calibration_range = open_value - closed_value
    if abs(calibration_range) <= 1e-6:
        return None
    return float(np.clip(
        (raw_value - closed_value) / calibration_range,
        0.0,
        1.0,
    ))


def calculate_finger_targets(openness, robot_model, reversed_direction=False):
    if robot_model.n < 7:
        return None, None

    effective = 1.0 - openness if reversed_direction else openness

    left_target = float(np.interp(
        effective,
        [0.0, 1.0],
        [robot_model.qlim[0, 5], robot_model.qlim[1, 5]],
    ))
    right_target = float(np.interp(
        effective,
        [0.0, 1.0],
        [robot_model.qlim[1, 6], robot_model.qlim[0, 6]],
    ))
    return left_target, right_target


def calculate_raw_arm_angles(data, target_arm):
    upper_arm_body = data.get("upper_arm_body")
    shoulder_world = data.get("shoulder_world")
    elbow_world = data.get("elbow_world")
    wrist_world = data.get("wrist_world")

    if (
        upper_arm_body is None
        or shoulder_world is None
        or elbow_world is None
        or wrist_world is None
        or not data.get("torso_reference_valid", False)
    ):
        return None

    upper_arm_body = np.asarray(upper_arm_body, dtype=np.float64)
    shoulder_world = np.asarray(shoulder_world, dtype=np.float64)
    elbow_world = np.asarray(elbow_world, dtype=np.float64)
    wrist_world = np.asarray(wrist_world, dtype=np.float64)

    side_sign = 1.0 if target_arm.lower() == "right" else -1.0
    lateral = side_sign * upper_arm_body[0]
    vertical_down = -upper_arm_body[1]

    shoulder_lateral = float(np.arctan2(lateral, vertical_down))
    shoulder_plane = float(np.arctan2(
        upper_arm_body[2],
        lateral,
    ))

    elbow_to_shoulder = shoulder_world - elbow_world
    elbow_to_wrist = wrist_world - elbow_world
    internal_angle = safe_angle_between(elbow_to_shoulder, elbow_to_wrist)
    if internal_angle is None:
        return None

    elbow_flexion = float(np.clip(np.pi - internal_angle, 0.0, np.pi))

    return {
        "shoulder_lateral": shoulder_lateral,
        "shoulder_plane": shoulder_plane,
        "elbow_flexion": elbow_flexion,
    }


def median_angles(samples):
    return {
        key: float(np.median([sample[key] for sample in samples]))
        for key in samples[0]
    }


def validate_arm_calibration(neutral, horizontal, elbow_90):
    if neutral is None or horizontal is None or elbow_90 is None:
        return False, None, None

    shoulder_range_deg = abs(np.degrees(
        horizontal["shoulder_lateral"]
        - neutral["shoulder_lateral"]
    ))
    elbow_range_deg = abs(np.degrees(
        elbow_90["elbow_flexion"]
        - horizontal["elbow_flexion"]
    ))

    valid = (
        shoulder_range_deg
        >= CONFIG["MIN_SHOULDER_CALIBRATION_RANGE_DEG"]
        and elbow_range_deg
        >= CONFIG["MIN_ELBOW_CALIBRATION_RANGE_DEG"]
    )
    return valid, float(shoulder_range_deg), float(elbow_range_deg)


def calculate_dynamic_elbow_zero(
    current_shoulder,
    neutral_shoulder,
    horizontal_shoulder,
    neutral_elbow,
    horizontal_elbow,
):
    shoulder_range = horizontal_shoulder - neutral_shoulder
    if abs(shoulder_range) <= 1e-6:
        return None, None

    progress = float(np.clip(
        (current_shoulder - neutral_shoulder) / shoulder_range,
        0.0,
        1.0,
    ))
    dynamic_zero = float(np.interp(
        progress,
        [0.0, 1.0],
        [neutral_elbow, horizontal_elbow],
    ))
    return dynamic_zero, progress


def calculate_calibrated_arm_targets(
    raw_angles,
    neutral,
    horizontal,
    elbow_90,
):
    """Return shoulder plane, shoulder elevation and elbow targets."""
    shoulder_range = (
        horizontal["shoulder_lateral"]
        - neutral["shoulder_lateral"]
    )
    if abs(shoulder_range) <= 1e-6:
        return None

    shoulder_normalized = float(np.clip(
        (
            raw_angles["shoulder_lateral"]
            - neutral["shoulder_lateral"]
        ) / shoulder_range,
        0.0,
        1.0,
    ))
    shoulder_elevation = shoulder_normalized * np.radians(
        CONFIG["SHOULDER_MAX_DEG"]
    )

    elbow_zero, shoulder_progress = calculate_dynamic_elbow_zero(
        raw_angles["shoulder_lateral"],
        neutral["shoulder_lateral"],
        horizontal["shoulder_lateral"],
        neutral["elbow_flexion"],
        horizontal["elbow_flexion"],
    )
    if elbow_zero is None:
        return None

    elbow_range = (
        elbow_90["elbow_flexion"]
        - horizontal["elbow_flexion"]
    )
    if abs(elbow_range) <= 1e-6:
        return None

    elbow_normalized = float(np.clip(
        (raw_angles["elbow_flexion"] - elbow_zero) / elbow_range,
        0.0,
        1.0,
    ))
    elbow_target = elbow_normalized * np.radians(CONFIG["ELBOW_MAX_DEG"])

    # Plane angle is only reliable once the arm is sufficiently elevated.
    shoulder_plane = raw_angles["shoulder_plane"]

    return {
        "shoulder_plane": shoulder_plane,
        "shoulder_elevation": shoulder_elevation,
        "elbow": elbow_target,
        "shoulder_progress": shoulder_progress,
        "dynamic_elbow_zero": elbow_zero,
    }


def apply_dead_zone(value, dead_zone_degrees):
    if value is None:
        return None
    if abs(value) < np.radians(dead_zone_degrees):
        return 0.0
    return float(value)


def put_text(image, text, y, color=(255, 255, 255), scale=0.50, thickness=2):
    cv2.putText(
        image,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
    )


# ============================================================
# VALIDAR ARCHIVOS Y ROBOT
# ============================================================

for required_path, description in (
    (URDF_PATH, "archivo URDF"),
    (POSE_MODEL, "modelo Pose Landmarker"),
    (HAND_MODEL, "modelo Hand Landmarker"),
):
    if not required_path.is_file():
        raise FileNotFoundError(f"No se encontro el {description}:\n{required_path}")

robot = rtb.ERobot.URDF(str(URDF_PATH))
print(robot)

if robot.n < 7:
    raise RuntimeError(
        "El robot necesita al menos 7 joints: "
        "cinco del brazo y dos de la garra."
    )
if robot.qlim is None or robot.qlim.shape != (2, robot.n):
    raise RuntimeError("El URDF no contiene limites articulares validos.")
if not np.all(np.isfinite(robot.qlim[:, 5:7])):
    raise RuntimeError("Los joints 6 y 7 no tienen limites finitos.")

print("=" * 70)
print("Limites articulares")

for joint_index in range(robot.n):
    unit = "m" if joint_index in (5, 6) else "rad"
    print(
        f"Joint {joint_index + 1}: "
        f"{robot.qlim[0, joint_index]:+.4f} a "
        f"{robot.qlim[1, joint_index]:+.4f} {unit}"
    )

print("=" * 70)


# ============================================================
# INICIALIZACION
# ============================================================

env = swift.Swift()
env.launch(realtime=True)
env.add(robot)

cap = open_camera(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["CAMERA_WIDTH"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["CAMERA_HEIGHT"])
cap.set(cv2.CAP_PROP_FPS, CONFIG["CAMERA_FPS"])
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

pose_detector, hand_detector = initialize_detectors(
    POSE_MODEL,
    HAND_MODEL,
    CONFIG,
)
processor = ArmHandProcessor(
    pose_detector=pose_detector,
    hand_detector=hand_detector,
    target_arm=TARGET_ARM,
    config=CONFIG,
    use_smoothing=True,
)

program_start_time = time.perf_counter()
last_timestamp_ms = -1
last_print_time = 0.0
last_joint_diagnostic_time = 0.0
fps_counter = 0
fps_display = 0.0
fps_start_time = time.perf_counter()

last_valid_q = np.zeros(robot.n, dtype=np.float64)
last_valid_q = np.clip(last_valid_q, robot.qlim[0, :], robot.qlim[1, :])
robot.q = last_valid_q.copy()
lost_frame_count = 0

# Arm calibration states
arm_calibration_mode = None
arm_calibration_samples = []
arm_neutral = None
arm_horizontal = None
arm_elbow_90 = None
arm_calibrated = False
arm_calibration_message = "Brazo sin calibrar | C-H-E"
shoulder_plane_smoothed = None
shoulder_elevation_smoothed = None
elbow_smoothed = None
last_shoulder_plane = 0.0

# Gripper calibration states
gripper_calibration_mode = None
gripper_calibration_samples = []
gripper_closed_raw = None
gripper_open_raw = None
gripper_calibrated = False
gripper_message = "Garra sin calibrar | F cerrada | O abierta"
gripper_openness_smoothed = None
left_finger_target = None
right_finger_target = None


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

try:
    while True:
        frame_start = time.perf_counter()
        success, frame = cap.read()
        if not success or frame is None:
            print("ERROR: No se pudo obtener una imagen de la camara.")
            break

        fps_counter += 1
        now = time.perf_counter()
        fps_elapsed = now - fps_start_time
        if fps_elapsed >= 1.0:
            fps_display = fps_counter / fps_elapsed
            fps_counter = 0
            fps_start_time = now

        timestamp_ms = int((now - program_start_time) * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        processing_frame = cv2.resize(
            frame,
            (CONFIG["PROCESSING_WIDTH"], CONFIG["PROCESSING_HEIGHT"]),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )

        vision_start = time.perf_counter()
        data = processor.process_frame(
            mp_image=mp_image,
            frame=frame,
            timestamp_ms=timestamp_ms,
        )
        vision_time_ms = (time.perf_counter() - vision_start) * 1000.0

        raw_arm = calculate_raw_arm_angles(data, TARGET_ARM)
        raw_hand = data.get("raw_hand_openness")

        # ----------------------------------------------------
        # CALIBRACION DEL BRAZO
        # ----------------------------------------------------

        if (
            arm_calibration_mode is not None
            and raw_arm is not None
            and data.get("arm_detected_current", False)
            and data.get("torso_detected_current", False)
        ):
            arm_calibration_samples.append(raw_arm)
            count = len(arm_calibration_samples)
            required = CONFIG["ARM_CALIBRATION_SAMPLES"]
            names = {
                "neutral": "NEUTRAL",
                "horizontal": "HORIZONTAL",
                "elbow90": "CODO 90",
            }
            arm_calibration_message = (
                f"Calibrando {names[arm_calibration_mode]}: "
                f"{count}/{required}"
            )

            if count >= required:
                result = median_angles(arm_calibration_samples)
                if arm_calibration_mode == "neutral":
                    arm_neutral = result
                    arm_calibration_message = "Neutral registrado"
                elif arm_calibration_mode == "horizontal":
                    arm_horizontal = result
                    arm_calibration_message = "Horizontal registrado"
                else:
                    arm_elbow_90 = result
                    arm_calibration_message = "Codo 90 registrado"

                arm_calibration_mode = None
                arm_calibration_samples = []

                (
                    arm_calibrated,
                    shoulder_range_deg,
                    elbow_range_deg,
                ) = validate_arm_calibration(
                    arm_neutral,
                    arm_horizontal,
                    arm_elbow_90,
                )

                if arm_calibrated:
                    arm_calibration_message = (
                        "Brazo calibrado | "
                        f"Hombro: {shoulder_range_deg:.1f} | "
                        f"Codo: {elbow_range_deg:.1f}"
                    )
                    shoulder_plane_smoothed = None
                    shoulder_elevation_smoothed = None
                    elbow_smoothed = None
                    last_shoulder_plane = arm_horizontal["shoulder_plane"]

        # ----------------------------------------------------
        # CALIBRACION DE LA GARRA
        # ----------------------------------------------------

        if (
            gripper_calibration_mode is not None
            and raw_hand is not None
            and data.get("hand_detected_current", False)
        ):
            gripper_calibration_samples.append(float(raw_hand))
            count = len(gripper_calibration_samples)
            required = CONFIG["GRIPPER_CALIBRATION_SAMPLES"]
            name = "CERRADA" if gripper_calibration_mode == "closed" else "ABIERTA"
            gripper_message = f"Calibrando {name}: {count}/{required}"

            if count >= required:
                value = float(np.median(gripper_calibration_samples))
                if gripper_calibration_mode == "closed":
                    gripper_closed_raw = value
                else:
                    gripper_open_raw = value

                gripper_calibration_mode = None
                gripper_calibration_samples = []

                if (
                    gripper_closed_raw is not None
                    and gripper_open_raw is not None
                    and (
                        gripper_open_raw - gripper_closed_raw
                        >= CONFIG["GRIPPER_MIN_CALIBRATION_RANGE"]
                    )
                ):
                    gripper_calibrated = True
                    gripper_openness_smoothed = None
                    gripper_message = (
                        "Garra calibrada | "
                        f"C: {gripper_closed_raw:.3f} | "
                        f"A: {gripper_open_raw:.3f}"
                    )
                elif (
                    gripper_closed_raw is not None
                    and gripper_open_raw is not None
                ):
                    gripper_calibrated = False
                    gripper_message = "Calibracion de garra invalida"

        # ----------------------------------------------------
        # OBJETIVOS DEL BRAZO
        # ----------------------------------------------------

        arm_targets = None
        if arm_calibrated and raw_arm is not None:
            arm_targets = calculate_calibrated_arm_targets(
                raw_arm,
                arm_neutral,
                arm_horizontal,
                arm_elbow_90,
            )

        if arm_targets is not None:
            elevation = apply_dead_zone(
                arm_targets["shoulder_elevation"],
                CONFIG["ARM_DEAD_ZONE_DEG"],
            )
            elbow_target = apply_dead_zone(
                arm_targets["elbow"],
                CONFIG["ARM_DEAD_ZONE_DEG"],
            )

            # Freeze plane angle near the hanging neutral position.
            if np.degrees(elevation) >= CONFIG["SHOULDER_PLANE_UPDATE_MIN_DEG"]:
                last_shoulder_plane = arm_targets["shoulder_plane"]

            shoulder_plane_smoothed = smooth_scalar(
                shoulder_plane_smoothed,
                last_shoulder_plane,
                CONFIG["ARM_CONTROL_SMOOTHING_ALPHA"],
            )
            shoulder_elevation_smoothed = smooth_scalar(
                shoulder_elevation_smoothed,
                elevation,
                CONFIG["ARM_CONTROL_SMOOTHING_ALPHA"],
            )
            elbow_smoothed = smooth_scalar(
                elbow_smoothed,
                elbow_target,
                CONFIG["ARM_CONTROL_SMOOTHING_ALPHA"],
            )

        # ----------------------------------------------------
        # OBJETIVOS DE LA GARRA
        # ----------------------------------------------------

        if gripper_calibrated and raw_hand is not None:
            openness = normalize_gripper_openness(
                raw_hand,
                gripper_closed_raw,
                gripper_open_raw,
            )
            gripper_openness_smoothed = smooth_scalar(
                gripper_openness_smoothed,
                openness,
                CONFIG["GRIPPER_SMOOTHING_ALPHA"],
            )
            left_finger_target, right_finger_target = calculate_finger_targets(
                gripper_openness_smoothed,
                robot,
                CONFIG["GRIPPER_REVERSED"],
            )

        # ----------------------------------------------------
        # ACTUALIZAR ROBOT
        # ----------------------------------------------------

        tracking_valid = False
        q_target = last_valid_q.copy()
        q_before_clip = None

        if data.get("arm_detected", False) and data.get("torso_reference_valid", False):
            if (
                arm_calibrated
                and shoulder_plane_smoothed is not None
                and shoulder_elevation_smoothed is not None
                and elbow_smoothed is not None
            ):
                # q0 defines the flexion plane. q1 raises the hanging arm.
                # --------------------------------------------
                # JOINT 1: PLANO DEL HOMBRO
                # --------------------------------------------
                shoulder_plane_offset = np.radians(
                    CONFIG["SHOULDER_PLANE_OFFSET_DEG"]
                )

                q_target[0] = (
                    CONFIG["SHOULDER_PLANE_SIGN"]
                    * shoulder_plane_smoothed
                    + shoulder_plane_offset
                )

                # Normalizar el ángulo al intervalo -pi a +pi.
                q_target[0] = (
                    q_target[0] + np.pi
                ) % (
                    2.0 * np.pi
                ) - np.pi
                
                q_target[1] = (
                    CONFIG["SHOULDER_ELEVATION_SIGN"]
                    * shoulder_elevation_smoothed
                )
                q_target[2] = CONFIG["ELBOW_SIGN"] * elbow_smoothed

            if (
                gripper_calibrated
                and left_finger_target is not None
                and right_finger_target is not None
                and data.get("hand_detected", False)
            ):
                q_target[5] = left_finger_target
                q_target[6] = right_finger_target

            q_before_clip = q_target.copy()
            q_target = np.clip(q_target, robot.qlim[0, :], robot.qlim[1, :])

            if (
                arm_calibrated
                and now - last_joint_diagnostic_time
                >= CONFIG["PRINT_INTERVAL_SECONDS"]
            ):
                clipped = not np.allclose(
                    q_before_clip[:3],
                    q_target[:3],
                )
                print(
                    "Diagnostico joints brazo | "
                    f"antes=[{q_before_clip[0]:+.4f}, "
                    f"{q_before_clip[1]:+.4f}, "
                    f"{q_before_clip[2]:+.4f}] rad | "
                    f"despues=[{q_target[0]:+.4f}, "
                    f"{q_target[1]:+.4f}, "
                    f"{q_target[2]:+.4f}] rad | "
                    f"recortado={'SI' if clipped else 'NO'}"
                )
                last_joint_diagnostic_time = now

            last_valid_q = q_target.copy()
            lost_frame_count = 0
            tracking_valid = True

        if not tracking_valid:
            lost_frame_count += 1
            q_target = last_valid_q.copy()

        if (
            arm_calibrated
            and shoulder_plane_smoothed is not None
            and shoulder_elevation_smoothed is not None
            and elbow_smoothed is not None
            and tracking_valid
        ):
            arm_control_status = "ACTIVO"
        elif arm_neutral is None:
            arm_control_status = "ESPERANDO C"
        elif arm_horizontal is None:
            arm_control_status = "ESPERANDO H"
        elif arm_elbow_90 is None:
            arm_control_status = "ESPERANDO E"
        elif not arm_calibrated:
            arm_control_status = "RANGO INVALIDO"
        elif not (
            data.get("arm_detected", False)
            and data.get("torso_reference_valid", False)
        ):
            arm_control_status = "TRACKING INVALIDO"
        elif (
            arm_targets is None
            or shoulder_plane_smoothed is None
            or shoulder_elevation_smoothed is None
            or elbow_smoothed is None
        ):
            arm_control_status = "OBJETIVOS NO CALCULADOS"
        else:
            arm_control_status = "TRACKING INVALIDO"

        swift_start = time.perf_counter()
        robot.q = q_target
        env.step(CONFIG["SWIFT_STEP_SECONDS"])
        swift_time_ms = (time.perf_counter() - swift_start) * 1000.0

        if (
            CONFIG["DEBUG_PRINT_COORDINATES"]
            and data.get("arm_detected", False)
            and data.get("hand_detected", False)
            and data.get("hand") is not None
            and now - last_print_time >= CONFIG["PRINT_INTERVAL_SECONDS"]
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

        total_time_ms = (time.perf_counter() - frame_start) * 1000.0

        # ----------------------------------------------------
        # INTERFAZ
        # ----------------------------------------------------

        display = cv2.flip(frame, 1)
        display_height, display_width, _ = display.shape

        put_text(display, f"FPS: {fps_display:.1f}", 30, scale=0.65)
        put_text(
            display,
            "TORSO Y BRAZO VALIDOS" if (
                data.get("arm_detected", False)
                and data.get("torso_reference_valid", False)
            ) else "TRACKING INCOMPLETO",
            60,
            (0, 255, 0) if tracking_valid else (0, 165, 255),
        )
        put_text(
            display,
            arm_calibration_message,
            90,
            (0, 255, 0) if arm_calibrated else (0, 165, 255),
        )
        put_text(
            display,
            gripper_message,
            120,
            (0, 255, 0) if gripper_calibrated else (0, 165, 255),
        )
        put_text(
            display,
            f"Estado control brazo: {arm_control_status}",
            270,
            (0, 255, 0) if arm_control_status == "ACTIVO" else (0, 165, 255),
            0.50,
        )

        if raw_arm is not None:
            put_text(
                display,
                (
                    "Crudos | Hombro: "
                    f"{np.degrees(raw_arm['shoulder_lateral']):.1f} | "
                    f"Codo: {np.degrees(raw_arm['elbow_flexion']):.1f}"
                ),
                150,
                (255, 255, 0),
            )

        if shoulder_elevation_smoothed is not None and elbow_smoothed is not None:
            put_text(
                display,
                (
                    "Robot | Plano: "
                    f"{np.degrees(shoulder_plane_smoothed):.1f} | "
                    "Hombro: "
                    f"{np.degrees(shoulder_elevation_smoothed):.1f} | "
                    f"Codo: {np.degrees(elbow_smoothed):.1f}"
                ),
                180,
                (0, 255, 0),
            )

        if gripper_openness_smoothed is not None:
            put_text(
                display,
                f"Garra: {gripper_openness_smoothed * 100.0:.1f} %",
                210,
                (0, 255, 0),
            )

        if CONFIG["SHOW_PROCESSING_TIME"]:
            put_text(
                display,
                (
                    f"Pose {data['pose_time_ms']:.1f} ms | "
                    f"Hand {data['hand_time_ms']:.1f} ms | "
                    f"Vision {vision_time_ms:.1f} ms | "
                    f"Swift {swift_time_ms:.1f} ms | "
                    f"Total {total_time_ms:.1f} ms"
                ),
                240,
                (255, 255, 255),
                0.45,
                1,
            )

        put_text(
            display,
            "Brazo: C neutral | H horizontal | E codo 90 | X reiniciar",
            display_height - 70,
            (220, 220, 220),
            0.45,
            1,
        )
        put_text(
            display,
            "Garra: F cerrada | O abierta | R reiniciar | ESC salir",
            display_height - 40,
            (220, 220, 220),
            0.45,
            1,
        )

        cv2.imshow("MediaPipe - Robot calibrado", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        # Arm controls
        elif key in (ord("c"), ord("C")):
            if raw_arm is None:
                arm_calibration_message = "No hay brazo/torso valido"
            else:
                arm_calibration_mode = "neutral"
                arm_calibration_samples = []
                arm_calibration_message = "Manten brazo abajo y recto"
        elif key in (ord("h"), ord("H")):
            if raw_arm is None:
                arm_calibration_message = "No hay brazo/torso valido"
            else:
                arm_calibration_mode = "horizontal"
                arm_calibration_samples = []
                arm_calibration_message = "Manten brazo horizontal y recto"
        elif key in (ord("e"), ord("E")):
            if raw_arm is None:
                arm_calibration_message = "No hay brazo/torso valido"
            else:
                arm_calibration_mode = "elbow90"
                arm_calibration_samples = []
                arm_calibration_message = "Brazo horizontal y muneca arriba"
        elif key in (ord("x"), ord("X")):
            arm_calibration_mode = None
            arm_calibration_samples = []
            arm_neutral = None
            arm_horizontal = None
            arm_elbow_90 = None
            arm_calibrated = False
            shoulder_plane_smoothed = None
            shoulder_elevation_smoothed = None
            elbow_smoothed = None
            arm_calibration_message = "Calibracion de brazo reiniciada"

        # Gripper controls
        elif key in (ord("f"), ord("F")):
            if raw_hand is None:
                gripper_message = "No se detecta la mano"
            else:
                gripper_calibration_mode = "closed"
                gripper_calibration_samples = []
                gripper_message = "Manten la mano cerrada"
        elif key in (ord("o"), ord("O")):
            if raw_hand is None:
                gripper_message = "No se detecta la mano"
            else:
                gripper_calibration_mode = "open"
                gripper_calibration_samples = []
                gripper_message = "Manten la mano abierta"
        elif key in (ord("r"), ord("R")):
            gripper_calibration_mode = None
            gripper_calibration_samples = []
            gripper_closed_raw = None
            gripper_open_raw = None
            gripper_calibrated = False
            gripper_openness_smoothed = None
            left_finger_target = None
            right_finger_target = None
            gripper_message = "Calibracion de garra reiniciada"

except KeyboardInterrupt:
    print("Programa detenido con Ctrl + C.")
finally:
    print("Liberando recursos...")
    cap.release()
    pose_detector.close()
    hand_detector.close()
    cv2.destroyAllWindows()
    try:
        env.close()
    except Exception as close_error:
        print(f"Swift no pudo cerrarse automaticamente: {close_error}")
    print("Recursos liberados correctamente.")

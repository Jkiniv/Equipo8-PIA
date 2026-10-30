"""Simulacion en tiempo real de un robot URDF de 7 joints controlado con MediaPipe.

Controles en la ventana de OpenCV:
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

BASE_DIRECTORY = Path(__file__).parent
URDF_PATH = BASE_DIRECTORY / "hachi.urdf"
POSE_MODEL = BASE_DIRECTORY / "modelos" / "pose_landmarker_full.task"
HAND_MODEL = BASE_DIRECTORY / "modelos" / "hand_landmarker.task"
CAMERA_INDEX = 0
TARGET_ARM = "right"

CONFIG = {
    "POSE_DETECTION_CONFIDENCE": 0.2,
    "POSE_PRESENCE_CONFIDENCE": 0.2,
    "POSE_TRACKING_CONFIDENCE": 0.2,
    "HAND_DETECTION_CONFIDENCE": 0.2,
    "HAND_PRESENCE_CONFIDENCE": 0.2,
    "HAND_TRACKING_CONFIDENCE": 0.2,
    "NUM_HANDS": 2,
    "CAMERA_WIDTH": 1280,
    "CAMERA_HEIGHT": 720,
    "CAMERA_FPS": 30,
    "PROCESSING_WIDTH": 640,
    "PROCESSING_HEIGHT": 360,
    "SMOOTHING_ALPHA": 0.35,
    "MAX_HAND_WRIST_DISTANCE": 0.30,
    "LANDMARK_VISIBILITY_THRESHOLD": 0.20,
    "ARM_LOST_FRAME_TOLERANCE": 5,
    "HAND_LOST_FRAME_TOLERANCE": 5,
    "LOST_FRAME_TOLERANCE": 5,
    "DRAW_ARM": True,
    "DRAW_HAND": True,
    "DRAW_FULL_HAND": False,
    "TORSO_LOST_FRAME_TOLERANCE": 5,

    "DRAW_TORSO": True,
    "DRAW_TORSO_AXES": True,

    # Longitud visual de los ejes normalizados.
    "TORSO_AXIS_LENGTH": 0.12,
    "DEBUG_PRINT_COORDINATES": False,
    "PRINT_INTERVAL_SECONDS": 1.0,
    "SHOW_PROCESSING_TIME": True,
    "SHOW_RAW_HAND_OPENNESS": True,
    "GRIPPER_CALIBRATION_SAMPLES": 30,
    "GRIPPER_MIN_CALIBRATION_RANGE": 0.20,
    "GRIPPER_SMOOTHING_ALPHA": 0.30,
    "GRIPPER_REVERSED": False,
    "SWIFT_STEP_SECONDS": 0.01,
}


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


def normalize_gripper_openness(raw_value, closed_value, open_value):
    calibration_range = open_value - closed_value
    if abs(calibration_range) <= 1e-6:
        return None
    return float(np.clip((raw_value - closed_value) / calibration_range, 0.0, 1.0))


def smooth_scalar(previous_value, current_value, alpha):
    if previous_value is None:
        return float(current_value)
    return float(alpha * current_value + (1.0 - alpha) * previous_value)


def validate_calibration(closed_value, open_value, minimum_range):
    if closed_value is None or open_value is None:
        return False, "Calibracion incompleta"
    difference = open_value - closed_value
    if difference < 0:
        return False, "Calibracion invertida | Repite F y O"
    if difference < minimum_range:
        return False, "Rango insuficiente | Repite F y O"
    return True, "Calibracion valida"


def calculate_finger_targets(openness, robot_model, reversed_direction=False):
    """Mapea 0..1 a q5 y q6 de la garra paralela."""
    if robot_model.n < 7:
        return None, None

    effective_openness = 1.0 - openness if reversed_direction else openness
    left_target = float(np.interp(
        effective_openness,
        [0.0, 1.0],
        [robot_model.qlim[0, 5], robot_model.qlim[1, 5]],
    ))
    right_target = float(np.interp(
        effective_openness,
        [0.0, 1.0],
        [robot_model.qlim[1, 6], robot_model.qlim[0, 6]],
    ))
    return left_target, right_target


def put_text(image, text, y, color=(255, 255, 255), scale=0.55, thickness=2):
    cv2.putText(image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)


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
        "El robot necesita al menos 7 articulaciones: "
        "cinco para el brazo y dos para la garra."
    )
if robot.qlim is None or robot.qlim.shape != (2, robot.n):
    raise RuntimeError("El URDF no contiene limites articulares validos.")
if not np.all(np.isfinite(robot.qlim[:, 5:7])):
    raise RuntimeError("Los joints 6 y 7 no tienen limites finitos para la garra.")

print("Limites articulares detectados:")
for joint_index in range(robot.n):
    unit = "m" if joint_index in (5, 6) else "rad"
    print(
        f"Joint {joint_index + 1}: {robot.qlim[0, joint_index]:.4f} "
        f"a {robot.qlim[1, joint_index]:.4f} {unit}"
    )

env = swift.Swift()
env.launch(realtime=True)
env.add(robot)

cap = open_camera(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["CAMERA_WIDTH"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["CAMERA_HEIGHT"])
cap.set(cv2.CAP_PROP_FPS, CONFIG["CAMERA_FPS"])
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

pose_detector, hand_detector = initialize_detectors(POSE_MODEL, HAND_MODEL, CONFIG)
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
fps_counter = 0
fps_display = 0.0
fps_start_time = time.perf_counter()

last_valid_q = np.zeros(robot.n, dtype=np.float64)
last_valid_q = np.clip(last_valid_q, robot.qlim[0, :], robot.qlim[1, :])
robot.q = last_valid_q.copy()
lost_frame_count = 0

gripper_calibration_mode = None
gripper_calibration_values = []
gripper_closed_raw = None
gripper_open_raw = None
gripper_calibrated = False
gripper_calibration_message = "Sin calibrar | F: cerrada | O: abierta"
gripper_openness = None
gripper_openness_smoothed = None
left_finger_target = None
right_finger_target = None

print("F: cerrada | O: abierta | R: reiniciar | ESC: salir")

try:
    while True:
        frame_start_time = time.perf_counter()
        success, frame = cap.read()
        if not success or frame is None:
            print("ERROR: No se pudo obtener una imagen de la camara.")
            break

        fps_counter += 1
        current_time = time.perf_counter()
        fps_elapsed = current_time - fps_start_time
        if fps_elapsed >= 1.0:
            fps_display = fps_counter / fps_elapsed
            fps_counter = 0
            fps_start_time = current_time

        timestamp_ms = int((current_time - program_start_time) * 1000)
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

        vision_start_time = time.perf_counter()
        data = processor.process_frame(
            mp_image=mp_image,
            frame=frame,
            timestamp_ms=timestamp_ms,
        )
        vision_time_ms = (time.perf_counter() - vision_start_time) * 1000.0
        raw_hand_openness = data.get("raw_hand_openness")

        # Recolectar muestras para calibracion.
        if (
            gripper_calibration_mode is not None
            and raw_hand_openness is not None
            and data.get("hand_detected_current", False)
        ):
            gripper_calibration_values.append(float(raw_hand_openness))
            collected = len(gripper_calibration_values)
            required = CONFIG["GRIPPER_CALIBRATION_SAMPLES"]
            name = "CERRADA" if gripper_calibration_mode == "closed" else "ABIERTA"
            gripper_calibration_message = f"Calibrando {name}: {collected}/{required}"

            if collected >= required:
                calibrated_value = float(np.median(gripper_calibration_values))
                if gripper_calibration_mode == "closed":
                    gripper_closed_raw = calibrated_value
                else:
                    gripper_open_raw = calibrated_value

                gripper_calibration_mode = None
                gripper_calibration_values = []
                gripper_calibrated, validation_message = validate_calibration(
                    gripper_closed_raw,
                    gripper_open_raw,
                    CONFIG["GRIPPER_MIN_CALIBRATION_RANGE"],
                )

                if gripper_calibrated:
                    gripper_openness_smoothed = None
                    gripper_calibration_message = (
                        f"Calibracion COMPLETA | Cerrada: {gripper_closed_raw:.3f} | "
                        f"Abierta: {gripper_open_raw:.3f}"
                    )
                elif gripper_closed_raw is None or gripper_open_raw is None:
                    registered_name = "Cerrada" if gripper_closed_raw is not None else "Abierta"
                    registered_value = (
                        gripper_closed_raw if gripper_closed_raw is not None
                        else gripper_open_raw
                    )
                    gripper_calibration_message = (
                        f"{registered_name} registrada: {registered_value:.3f}"
                    )
                else:
                    gripper_calibration_message = validation_message
                    gripper_openness_smoothed = None
                    left_finger_target = None
                    right_finger_target = None

        # Calcular porcentaje y posiciones de ambos dedos.
        if (
            gripper_calibrated
            and raw_hand_openness is not None
            and gripper_closed_raw is not None
            and gripper_open_raw is not None
        ):
            gripper_openness = normalize_gripper_openness(
                raw_hand_openness,
                gripper_closed_raw,
                gripper_open_raw,
            )
            if gripper_openness is not None:
                gripper_openness_smoothed = smooth_scalar(
                    gripper_openness_smoothed,
                    gripper_openness,
                    CONFIG["GRIPPER_SMOOTHING_ALPHA"],
                )
                left_finger_target, right_finger_target = calculate_finger_targets(
                    gripper_openness_smoothed,
                    robot,
                    CONFIG["GRIPPER_REVERSED"],
                )
        else:
            gripper_openness = None
            left_finger_target = None
            right_finger_target = None

        # Calculo actual del brazo.
        tracking_valid = False
        q_target = last_valid_q.copy()

        if (
            data["arm_detected"]
            and data["shoulder"] is not None
            and data["elbow"] is not None
            and data["wrist"] is not None
        ):
            shoulder = np.asarray(data["shoulder"], dtype=np.float64)
            elbow = np.asarray(data["elbow"], dtype=np.float64)
            wrist = np.asarray(data["wrist"], dtype=np.float64)
            upper_arm_vector = elbow - shoulder
            forearm_vector = wrist - elbow
            upper_norm = np.linalg.norm(upper_arm_vector)
            forearm_norm = np.linalg.norm(forearm_vector)

            if upper_norm > 1e-6 and forearm_norm > 1e-6:
                q_target[0] = np.arctan2(upper_arm_vector[0], -upper_arm_vector[2])
                horizontal_distance = np.linalg.norm(
                    [upper_arm_vector[0], upper_arm_vector[2]]
                )
                q_target[1] = np.arctan2(upper_arm_vector[1], horizontal_distance)
                cosine_angle = np.dot(upper_arm_vector, forearm_vector) / (
                    upper_norm * forearm_norm
                )
                q_target[2] = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

                if (
                    gripper_calibrated
                    and left_finger_target is not None
                    and right_finger_target is not None
                    and data["hand_detected"]
                ):
                    q_target[5] = left_finger_target
                    q_target[6] = right_finger_target
                else:
                    q_target[5] = last_valid_q[5]
                    q_target[6] = last_valid_q[6]

                q_target = np.clip(q_target, robot.qlim[0, :], robot.qlim[1, :])
                last_valid_q = q_target.copy()
                lost_frame_count = 0
                tracking_valid = True

        if not tracking_valid:
            lost_frame_count += 1
            q_target = last_valid_q.copy()

        swift_start_time = time.perf_counter()
        robot.q = q_target
        env.step(CONFIG["SWIFT_STEP_SECONDS"])
        swift_time_ms = (time.perf_counter() - swift_start_time) * 1000.0

        if (
            CONFIG["DEBUG_PRINT_COORDINATES"]
            and data["arm_detected"]
            and data["hand_detected"]
            and data["hand"] is not None
            and current_time - last_print_time >= CONFIG["PRINT_INTERVAL_SECONDS"]
        ):
            print_robot_coordinates(
                timestamp_ms=timestamp_ms,
                target_arm=TARGET_ARM,
                shoulder=data["shoulder"],
                elbow=data["elbow"],
                wrist=data["wrist"],
                hand=data["hand"],
            )
            last_print_time = current_time

        total_time_ms = (time.perf_counter() - frame_start_time) * 1000.0
        display_frame = cv2.flip(frame, 1)
        display_height, display_width, _ = display_frame.shape

        put_text(display_frame, f"Brazo: {TARGET_ARM.upper()}", 35, scale=0.75)
        cv2.putText(
            display_frame,
            f"FPS: {fps_display:.1f}",
            (display_width - 170, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
        )
        put_text(
            display_frame,
            "BRAZO DETECTADO" if data["arm_detected"] else "BRAZO NO DETECTADO",
            70,
            (0, 255, 0) if data["arm_detected"] else (0, 0, 255),
            0.70,
        )
        put_text(
            display_frame,
            "MANO DETECTADA" if data["hand_detected"] else "MANO NO DETECTADA",
            105,
            (0, 255, 0) if data["hand_detected"] else (0, 0, 255),
            0.65,
        )
        put_text(
            display_frame,
            f"Inferencia: {CONFIG['PROCESSING_WIDTH']}x{CONFIG['PROCESSING_HEIGHT']}",
            140,
        )

        if CONFIG["SHOW_PROCESSING_TIME"]:
            put_text(display_frame, f"Pose: {data['pose_time_ms']:.1f} ms", 170)
            put_text(display_frame, f"Hand: {data['hand_time_ms']:.1f} ms", 200)
            put_text(display_frame, f"Postproceso: {data['postprocess_time_ms']:.1f} ms", 230)
            put_text(display_frame, f"Vision total: {vision_time_ms:.1f} ms", 260)
            put_text(display_frame, f"Swift: {swift_time_ms:.1f} ms", 290)
            put_text(display_frame, f"Total: {total_time_ms:.1f} ms", 320)

        put_text(
            display_frame,
            "TRACKING ACTIVO" if tracking_valid
            else f"MANTENIENDO POSICION | Frames perdidos: {lost_frame_count}",
            355,
            (0, 255, 0) if tracking_valid else (0, 165, 255),
        )

        if CONFIG["SHOW_RAW_HAND_OPENNESS"] and raw_hand_openness is not None:
            put_text(display_frame, f"Apertura cruda: {raw_hand_openness:.3f}",
                     385, (255, 255, 0))

        put_text(
            display_frame,
            gripper_calibration_message,
            415,
            (0, 255, 0) if gripper_calibrated else (0, 165, 255),
            0.50,
        )

        if gripper_calibrated and gripper_openness_smoothed is not None:
            put_text(
                display_frame,
                f"Apertura calibrada: {gripper_openness_smoothed * 100.0:.1f} %",
                445,
                (255, 255, 0),
            )

        if left_finger_target is not None and right_finger_target is not None:
            side_mm = abs(left_finger_target) * 1000.0
            extra_mm = (abs(left_finger_target) + abs(right_finger_target)) * 1000.0
            put_text(
                display_frame,
                f"Dedos: {side_mm:.1f} mm por lado | Apertura extra: {extra_mm:.1f} mm",
                475,
                (255, 255, 0),
            )

        gripper_control_active = (
            gripper_calibrated
            and left_finger_target is not None
            and right_finger_target is not None
            and data["hand_detected"]
        )
        if gripper_control_active:
            control_text, control_color = "Control de garra: ACTIVO", (0, 255, 0)
        elif not gripper_calibrated:
            control_text, control_color = (
                "Control de garra: ESPERANDO CALIBRACION", (0, 165, 255)
            )
        else:
            control_text, control_color = "Control de garra: MANO NO DETECTADA", (0, 0, 255)

        put_text(display_frame, control_text, 505, control_color, 0.50)
        put_text(display_frame, "F: cerrada | O: abierta | R: reiniciar",
                 535, (220, 220, 220), 0.50, 1)
        put_text(
            display_frame,
            f"Perdidos | Brazo: {data['arm_lost_frames']} Mano: {data['hand_lost_frames']}",
            565,
            (200, 200, 200),
            0.50,
            1,
        )
        put_text(display_frame, "ESC: salir", display_height - 20,
                 (200, 200, 200), 0.55, 1)

        cv2.imshow("MediaPipe - Control de robot", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("\nPrograma detenido por el usuario.")
            break
        elif key in (ord("f"), ord("F")):
            if raw_hand_openness is None:
                gripper_calibration_message = "No se detecta la mano para calibrar"
            else:
                gripper_calibration_mode = "closed"
                gripper_calibration_values = []
                gripper_calibration_message = "Manten la mano CERRADA..."
        elif key in (ord("o"), ord("O")):
            if raw_hand_openness is None:
                gripper_calibration_message = "No se detecta la mano para calibrar"
            else:
                gripper_calibration_mode = "open"
                gripper_calibration_values = []
                gripper_calibration_message = "Manten la mano ABIERTA..."
        elif key in (ord("r"), ord("R")):
            gripper_calibration_mode = None
            gripper_calibration_values = []
            gripper_closed_raw = None
            gripper_open_raw = None
            gripper_calibrated = False
            gripper_openness = None
            gripper_openness_smoothed = None
            left_finger_target = None
            right_finger_target = None
            gripper_calibration_message = (
                "Calibracion reiniciada | F: cerrada | O: abierta"
            )

except KeyboardInterrupt:
    print("\nPrograma detenido mediante Ctrl + C.")
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

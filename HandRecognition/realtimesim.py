"""Simulacion en tiempo real de un robot URDF controlado con MediaPipe.

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


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def open_camera(camera_index):
    """Abre la camara con un backend adecuado para Windows o Linux."""
    system_name = platform.system().lower()

    if system_name == "windows":
        print("Sistema operativo detectado: Windows")
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    elif system_name == "linux":
        print("Sistema operativo detectado: Linux")
        capture = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    else:
        print(f"Sistema operativo detectado: {platform.system()}")
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        print("El backend especifico fallo. Intentando modo automatico...")
        capture.release()
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        raise RuntimeError(
            "No se pudo abrir la camara. Verifica CAMERA_INDEX y que "
            "ninguna otra aplicacion este utilizando la camara."
        )

    return capture


def normalize_gripper_openness(raw_openness, closed_raw, open_raw):
    """Convierte la apertura cruda a un valor limitado entre 0 y 1."""
    calibration_range = open_raw - closed_raw
    if abs(calibration_range) <= 1e-6:
        return None

    value = (raw_openness - closed_raw) / calibration_range
    return float(np.clip(value, 0.0, 1.0))


def smooth_scalar(previous_value, current_value, alpha):
    """Suavizado exponencial para un valor escalar."""
    if previous_value is None:
        return float(current_value)
    return float(alpha * current_value + (1.0 - alpha) * previous_value)


def map_gripper_to_joint(openness, robot_model, reversed_direction=False):
    """Mapea una apertura 0..1 al rango del joint 6 del URDF."""
    if robot_model.n < 6:
        return None

    joint_minimum = float(robot_model.qlim[0, 5])
    joint_maximum = float(robot_model.qlim[1, 5])

    output_range = (
        [joint_maximum, joint_minimum]
        if reversed_direction
        else [joint_minimum, joint_maximum]
    )

    return float(np.interp(openness, [0.0, 1.0], output_range))


def validate_calibration(closed_raw, open_raw, minimum_range):
    """Valida que abierta y cerrada esten en el orden y rango esperados."""
    if closed_raw is None or open_raw is None:
        return False, "Calibracion incompleta"

    calibration_range = open_raw - closed_raw

    if calibration_range < 0:
        return False, "Calibracion invertida | Repite F y O"

    if calibration_range < minimum_range:
        return False, "Rango insuficiente | Repite F y O"

    return True, "Calibracion valida"


def put_text(image, text, y, color=(255, 255, 255), scale=0.55, thickness=2):
    """Dibuja una linea de texto en la columna izquierda."""
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
# VALIDAR ARCHIVOS Y CARGAR ROBOT
# ============================================================

for required_path, description in (
    (URDF_PATH, "archivo URDF"),
    (POSE_MODEL, "modelo Pose Landmarker"),
    (HAND_MODEL, "modelo Hand Landmarker"),
):
    if not required_path.is_file():
        raise FileNotFoundError(f"No se encontro el {description}:\n{required_path}")

print("=" * 70)
print(f"Cargando robot desde: {URDF_PATH}")
print("=" * 70)

robot = rtb.ERobot.URDF(str(URDF_PATH))
print(robot)

if robot.n < 6:
    raise RuntimeError(
        "El proyecto requiere un robot de al menos 6 articulaciones "
        "para hombro, codo, muneca y garra."
    )

if robot.qlim is None or robot.qlim.shape != (2, robot.n):
    raise RuntimeError("El URDF no contiene limites articulares validos.")

if not np.all(np.isfinite(robot.qlim[:, 5])):
    raise RuntimeError("El joint 6 no tiene limites finitos para la garra.")

print("Limites articulares detectados:")
for joint_index in range(robot.n):
    print(
        f"Joint {joint_index + 1}: "
        f"{robot.qlim[0, joint_index]:.4f} a "
        f"{robot.qlim[1, joint_index]:.4f} rad"
    )


# ============================================================
# INICIALIZAR SWIFT, CAMARA Y MEDIAPIPE
# ============================================================

print("=" * 70)
print("Iniciando simulador Swift...")
print("=" * 70)

env = swift.Swift()
env.launch(realtime=True)
env.add(robot)

cap = open_camera(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["CAMERA_WIDTH"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["CAMERA_HEIGHT"])
cap.set(cv2.CAP_PROP_FPS, CONFIG["CAMERA_FPS"])
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("=" * 70)
print(
    "Camara: "
    f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
    f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} | "
    f"{cap.get(cv2.CAP_PROP_FPS):.2f} FPS"
)
print(
    "Resolucion de inferencia: "
    f"{CONFIG['PROCESSING_WIDTH']} x {CONFIG['PROCESSING_HEIGHT']}"
)
print(f"Brazo objetivo: {TARGET_ARM.upper()}")
print("=" * 70)

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


# ============================================================
# ESTADOS DEL PROGRAMA
# ============================================================

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

# Calibracion de garra.
gripper_calibration_mode = None
gripper_calibration_values = []
gripper_closed_raw = None
gripper_open_raw = None
gripper_calibrated = False
gripper_calibration_message = "Sin calibrar | F: cerrada | O: abierta"
gripper_openness = None
gripper_openness_smoothed = None
gripper_joint_target = None

print("Presiona F para cerrada, O para abierta, R para reiniciar y ESC para salir.")


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

try:
    while True:
        frame_start_time = time.perf_counter()

        success, frame = cap.read()
        if not success or frame is None:
            print("ERROR: No se pudo obtener una imagen de la camara.")
            break

        # FPS y timestamp monotono.
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

        # Inferencia en resolucion reducida.
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

        # ----------------------------------------------------
        # CALIBRACION DE GARRA
        # ----------------------------------------------------

        raw_hand_openness = data.get("raw_hand_openness")

        if (
            gripper_calibration_mode is not None
            and raw_hand_openness is not None
            and data.get("hand_detected_current", False)
        ):
            gripper_calibration_values.append(float(raw_hand_openness))
            collected = len(gripper_calibration_values)
            required = CONFIG["GRIPPER_CALIBRATION_SAMPLES"]
            pose_name = "CERRADA" if gripper_calibration_mode == "closed" else "ABIERTA"
            gripper_calibration_message = f"Calibrando {pose_name}: {collected}/{required}"

            if collected >= required:
                calibration_value = float(np.median(gripper_calibration_values))

                if gripper_calibration_mode == "closed":
                    gripper_closed_raw = calibration_value
                else:
                    gripper_open_raw = calibration_value

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
                        "Calibracion COMPLETA | "
                        f"Cerrada: {gripper_closed_raw:.3f} | "
                        f"Abierta: {gripper_open_raw:.3f}"
                    )
                elif gripper_closed_raw is None or gripper_open_raw is None:
                    registered_name = "Cerrada" if gripper_closed_raw is not None else "Abierta"
                    registered_value = (
                        gripper_closed_raw
                        if gripper_closed_raw is not None
                        else gripper_open_raw
                    )
                    gripper_calibration_message = (
                        f"{registered_name} registrada: {registered_value:.3f}"
                    )
                else:
                    gripper_openness = None
                    gripper_openness_smoothed = None
                    gripper_joint_target = None
                    gripper_calibration_message = validation_message

        # Convertir apertura calibrada en objetivo del joint 6.
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
                gripper_joint_target = map_gripper_to_joint(
                    gripper_openness_smoothed,
                    robot,
                    CONFIG["GRIPPER_REVERSED"],
                )
        else:
            gripper_openness = None
            gripper_joint_target = None

        # ----------------------------------------------------
        # CALCULO ARTICULAR ACTUAL
        # ----------------------------------------------------

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
            upper_arm_norm = np.linalg.norm(upper_arm_vector)
            forearm_norm = np.linalg.norm(forearm_vector)

            if upper_arm_norm > 1e-6 and forearm_norm > 1e-6:
                q_target[0] = np.arctan2(
                    upper_arm_vector[0],
                    -upper_arm_vector[2],
                )

                horizontal_distance = np.linalg.norm(
                    [upper_arm_vector[0], upper_arm_vector[2]]
                )
                q_target[1] = np.arctan2(
                    upper_arm_vector[1],
                    horizontal_distance,
                )

                cosine_angle = np.dot(
                    upper_arm_vector,
                    forearm_vector,
                ) / (upper_arm_norm * forearm_norm)
                q_target[2] = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

                # Control proporcional de la garra despues de calibrar.
                if (
                    gripper_calibrated
                    and gripper_joint_target is not None
                    and data["hand_detected"]
                ):
                    q_target[5] = gripper_joint_target
                else:
                    q_target[5] = last_valid_q[5]

                q_target = np.clip(q_target, robot.qlim[0, :], robot.qlim[1, :])
                last_valid_q = q_target.copy()
                lost_frame_count = 0
                tracking_valid = True

        if not tracking_valid:
            lost_frame_count += 1
            q_target = last_valid_q.copy()

        # Swift.
        swift_start_time = time.perf_counter()
        robot.q = q_target
        env.step(CONFIG["SWIFT_STEP_SECONDS"])
        swift_time_ms = (time.perf_counter() - swift_start_time) * 1000.0

        # Impresion opcional.
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

        # ----------------------------------------------------
        # INTERFAZ
        # ----------------------------------------------------

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
            "TRACKING ACTIVO"
            if tracking_valid
            else f"MANTENIENDO POSICION | Frames perdidos: {lost_frame_count}",
            355,
            (0, 255, 0) if tracking_valid else (0, 165, 255),
        )

        if CONFIG["SHOW_RAW_HAND_OPENNESS"] and raw_hand_openness is not None:
            put_text(
                display_frame,
                f"Apertura cruda: {raw_hand_openness:.3f}",
                385,
                (255, 255, 0),
            )

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

        if gripper_joint_target is not None:
            put_text(
                display_frame,
                f"Joint 6 objetivo: {gripper_joint_target:.3f} rad",
                475,
                (255, 255, 0),
            )

        gripper_control_active = (
            gripper_calibrated
            and gripper_joint_target is not None
            and data["hand_detected"]
        )

        if gripper_control_active:
            gripper_control_text = "Control de garra: ACTIVO"
            gripper_control_color = (0, 255, 0)
        elif not gripper_calibrated:
            gripper_control_text = "Control de garra: ESPERANDO CALIBRACION"
            gripper_control_color = (0, 165, 255)
        else:
            gripper_control_text = "Control de garra: MANO NO DETECTADA"
            gripper_control_color = (0, 0, 255)

        put_text(
            display_frame,
            gripper_control_text,
            505,
            gripper_control_color,
            0.50,
        )
        put_text(
            display_frame,
            "F: cerrada | O: abierta | R: reiniciar",
            535,
            (220, 220, 220),
            0.50,
            1,
        )
        put_text(
            display_frame,
            f"Perdidos | Brazo: {data['arm_lost_frames']} Mano: {data['hand_lost_frames']}",
            565,
            (200, 200, 200),
            0.50,
            1,
        )
        put_text(
            display_frame,
            "ESC: salir",
            display_height - 20,
            (200, 200, 200),
            0.55,
            1,
        )

        cv2.imshow("MediaPipe - Control de robot", display_frame)

        # ----------------------------------------------------
        # CONTROLES
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("\nPrograma detenido por el usuario.")
            break

        if key in (ord("f"), ord("F")):
            if raw_hand_openness is None:
                gripper_calibration_message = "No se detecta la mano para calibrar"
            else:
                gripper_calibration_mode = "closed"
                gripper_calibration_values = []
                gripper_calibration_message = "Manten la mano CERRADA..."
                print("\nCalibracion de mano cerrada iniciada.")

        elif key in (ord("o"), ord("O")):
            if raw_hand_openness is None:
                gripper_calibration_message = "No se detecta la mano para calibrar"
            else:
                gripper_calibration_mode = "open"
                gripper_calibration_values = []
                gripper_calibration_message = "Manten la mano ABIERTA..."
                print("\nCalibracion de mano abierta iniciada.")

        elif key in (ord("r"), ord("R")):
            gripper_calibration_mode = None
            gripper_calibration_values = []
            gripper_closed_raw = None
            gripper_open_raw = None
            gripper_calibrated = False
            gripper_openness = None
            gripper_openness_smoothed = None
            gripper_joint_target = None
            gripper_calibration_message = (
                "Calibracion reiniciada | F: cerrada | O: abierta"
            )
            print("\nCalibracion de la garra reiniciada.")

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
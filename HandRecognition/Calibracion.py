"""Calibra el brazo humano usando torso y Pose World Landmarks.

C: postura neutral, brazo abajo y codo recto
H: brazo horizontal lateral y codo recto
E: brazo superior horizontal y muneca arriba, codo cerca de 90 grados
X: reiniciar calibracion
ESC: salir
"""

from pathlib import Path
import platform
import time

import cv2
import mediapipe as mp
import numpy as np

from detector import initialize_detectors
from processor import ArmHandProcessor


BASE_DIRECTORY = Path(__file__).parent
POSE_MODEL = BASE_DIRECTORY / "modelos" / "pose_landmarker_full.task"
HAND_MODEL = BASE_DIRECTORY / "modelos" / "hand_landmarker.task"
CAMERA_INDEX = 0
TARGET_ARM = "right"

CONFIG = {
    "POSE_DETECTION_CONFIDENCE": 0.4,
    "POSE_PRESENCE_CONFIDENCE": 0.4,
    "POSE_TRACKING_CONFIDENCE": 0.4,
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
    "LANDMARK_VISIBILITY_THRESHOLD": 0.30,
    "ARM_LOST_FRAME_TOLERANCE": 5,
    "HAND_LOST_FRAME_TOLERANCE": 5,
    "TORSO_LOST_FRAME_TOLERANCE": 5,
    "LOST_FRAME_TOLERANCE": 5,
    "DRAW_ARM": True,
    "DRAW_HAND": False,
    "DRAW_FULL_HAND": False,
    "DRAW_TORSO": True,
    "DRAW_TORSO_AXES": True,
    "TORSO_AXIS_LENGTH": 0.12,
    "ARM_CALIBRATION_SAMPLES": 30,
    "MIN_SHOULDER_CALIBRATION_RANGE_DEG": 35.0,
    "MIN_ELBOW_CALIBRATION_RANGE_DEG": 35.0,
    "ARM_PREVIEW_SMOOTHING_ALPHA": 0.25,
    "ARM_DEAD_ZONE_DEG": 3.0,
    "ARM_POSE_VALIDATION_TOLERANCE_DEG": 15.0,
    "SHOULDER_PREVIEW_MIN_DEG": 0.0,
    "SHOULDER_PREVIEW_MAX_DEG": 90.0,
    "ELBOW_PREVIEW_MIN_DEG": 0.0,
    "ELBOW_PREVIEW_MAX_DEG": 90.0,
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
    "ARM_PREVIEW_SMOOTHING_ALPHA",
    "ARM_DEAD_ZONE_DEG",
    "ARM_POSE_VALIDATION_TOLERANCE_DEG",
    "SHOULDER_PREVIEW_MIN_DEG",
    "SHOULDER_PREVIEW_MAX_DEG",
    "ELBOW_PREVIEW_MIN_DEG",
    "ELBOW_PREVIEW_MAX_DEG",
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


validate_config(CONFIG, REQUIRED_CONFIG_KEYS, "Calibracion.py")


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
    lateral_component = side_sign * upper_arm_body[0]
    vertical_down_component = -upper_arm_body[1]

    shoulder_lateral = float(np.arctan2(
        lateral_component,
        vertical_down_component,
    ))

    lateral_vertical_norm = np.linalg.norm(upper_arm_body[:2])
    shoulder_forward = float(np.arctan2(
        upper_arm_body[2],
        lateral_vertical_norm,
    ))

    elbow_to_shoulder = shoulder_world - elbow_world
    elbow_to_wrist = wrist_world - elbow_world
    internal_angle = safe_angle_between(elbow_to_shoulder, elbow_to_wrist)
    if internal_angle is None:
        return None

    elbow_flexion = float(np.clip(np.pi - internal_angle, 0.0, np.pi))

    return {
        "shoulder_lateral": shoulder_lateral,
        "shoulder_forward": shoulder_forward,
        "elbow_flexion": elbow_flexion,
    }


def median_angles(samples):
    return {
        key: float(np.median([sample[key] for sample in samples]))
        for key in samples[0]
    }


def degrees(value):
    return None if value is None else float(np.degrees(value))


def map_calibrated_value(raw, raw_zero, raw_reference, output_reference):
    denominator = raw_reference - raw_zero
    if abs(denominator) <= 1e-6:
        return None
    return float((raw - raw_zero) / denominator * output_reference)


def calculate_dynamic_elbow_zero(
    current_shoulder,
    neutral_shoulder,
    horizontal_shoulder,
    neutral_elbow,
    horizontal_elbow,
):
    """
    Calcula el valor crudo que representa un codo recto segun
    la elevacion actual del hombro.

    C:
        Brazo hacia abajo y codo recto.

    H:
        Brazo horizontal y codo recto.

    El cero del codo se interpola entre ambas referencias para
    compensar el sesgo de MediaPipe segun la orientacion del brazo.

    Devuelve:
        dynamic_elbow_zero:
            Valor crudo esperado para un codo recto en la
            elevacion actual del hombro.

        shoulder_progress:
            Progreso normalizado del hombro entre C y H.
            0.0 representa la postura neutral C.
            1.0 representa la postura horizontal H.
    """

    shoulder_range = (
        horizontal_shoulder
        - neutral_shoulder
    )

    if abs(shoulder_range) <= 1e-6:
        return None, None

    shoulder_progress = (
        current_shoulder
        - neutral_shoulder
    ) / shoulder_range

    shoulder_progress = float(
        np.clip(
            shoulder_progress,
            0.0,
            1.0,
        )
    )

    dynamic_elbow_zero = float(
        np.interp(
            shoulder_progress,
            [0.0, 1.0],
            [
                neutral_elbow,
                horizontal_elbow,
            ],
        )
    )

    return dynamic_elbow_zero, shoulder_progress


def smooth_scalar(
    previous_value,
    current_value,
    alpha,
):
    """
    Aplica suavizado exponencial a un valor escalar.
    """

    if current_value is None:
        return previous_value

    if previous_value is None:
        return float(current_value)

    return float(
        alpha * current_value
        + (1.0 - alpha) * previous_value
    )


def clamp_angle_degrees(
    angle_radians,
    minimum_degrees,
    maximum_degrees,
):
    """
    Limita un angulo en radianes utilizando limites
    especificados en grados.
    """

    if angle_radians is None:
        return None

    minimum_radians = np.radians(
        minimum_degrees
    )

    maximum_radians = np.radians(
        maximum_degrees
    )

    return float(
        np.clip(
            angle_radians,
            minimum_radians,
            maximum_radians,
        )
    )


def apply_dead_zone(
    angle_radians,
    dead_zone_degrees,
):
    """
    Convierte a cero los angulos pequenos dentro de la
    zona muerta.
    """

    if angle_radians is None:
        return None

    dead_zone_radians = np.radians(
        dead_zone_degrees
    )

    if abs(angle_radians) < dead_zone_radians:
        return 0.0

    return float(angle_radians)


def classify_calibrated_pose(
    shoulder_degrees,
    elbow_degrees,
    tolerance_degrees,
):
    """
    Clasifica aproximadamente la postura actual utilizando
    los previews calibrados.

    C:
        Hombro cerca de 0 grados.
        Codo cerca de 0 grados.

    H:
        Hombro cerca de 90 grados.
        Codo cerca de 0 grados.

    E:
        Hombro cerca de 90 grados.
        Codo cerca de 90 grados.
    """

    if (
        shoulder_degrees is None
        or elbow_degrees is None
    ):
        return "SIN DATOS"

    near_shoulder_zero = (
        abs(shoulder_degrees)
        <= tolerance_degrees
    )

    near_shoulder_horizontal = (
        abs(shoulder_degrees - 90.0)
        <= tolerance_degrees
    )

    near_elbow_zero = (
        abs(elbow_degrees)
        <= tolerance_degrees
    )

    near_elbow_ninety = (
        abs(elbow_degrees - 90.0)
        <= tolerance_degrees
    )

    if (
        near_shoulder_zero
        and near_elbow_zero
    ):
        return "C - NEUTRAL"

    if (
        near_shoulder_horizontal
        and near_elbow_zero
    ):
        return "H - HORIZONTAL RECTO"

    if (
        near_shoulder_horizontal
        and near_elbow_ninety
    ):
        return "E - CODO 90"

    return "INTERMEDIA"


def put_text(image, text, y, color=(255, 255, 255), scale=0.55, thickness=2):
    cv2.putText(
        image,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
    )


for path, description in (
    (POSE_MODEL, "modelo Pose Landmarker"),
    (HAND_MODEL, "modelo Hand Landmarker"),
):
    if not path.is_file():
        raise FileNotFoundError(f"No se encontro el {description}:\n{path}")

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
fps_counter = 0
fps_display = 0.0
fps_start_time = time.perf_counter()

calibration_mode = None
calibration_samples = []
neutral_calibration = None
horizontal_calibration = None
elbow_90_calibration = None
shoulder_preview_smoothed = None
elbow_preview_smoothed = None
current_pose_name = "SIN CALIBRAR"
calibration_was_valid = False
calibration_message = "C neutral | H horizontal recto | E muneca arriba | X reiniciar"

try:
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("ERROR: No se pudo obtener una imagen de la camara.")
            break

        fps_counter += 1
        now = time.perf_counter()
        elapsed = now - fps_start_time
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_start_time = now

        timestamp_ms = int((now - program_start_time) * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        small_frame = cv2.resize(
            frame,
            (CONFIG["PROCESSING_WIDTH"], CONFIG["PROCESSING_HEIGHT"]),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )

        data = processor.process_frame(
            mp_image=mp_image,
            frame=frame,
            timestamp_ms=timestamp_ms,
        )
        raw_angles = calculate_raw_arm_angles(data, TARGET_ARM)

        if (
            calibration_mode is not None
            and raw_angles is not None
            and data.get("arm_detected_current", False)
            and data.get("torso_detected_current", False)
        ):
            calibration_samples.append(raw_angles)
            count = len(calibration_samples)
            required = CONFIG["ARM_CALIBRATION_SAMPLES"]
            names = {
                "neutral": "NEUTRAL",
                "horizontal": "HORIZONTAL",
                "elbow90": "CODO 90",
            }
            calibration_message = (
                f"Calibrando {names[calibration_mode]}: {count}/{required}"
            )

            if count >= required:
                result = median_angles(calibration_samples)
                if calibration_mode == "neutral":
                    neutral_calibration = result
                    calibration_message = "Postura neutral registrada"
                elif calibration_mode == "horizontal":
                    horizontal_calibration = result
                    calibration_message = "Brazo horizontal recto registrado"
                else:
                    elbow_90_calibration = result
                    calibration_message = "Codo a 90 grados registrado"
                calibration_mode = None
                calibration_samples = []

        shoulder_preview = None
        elbow_preview = None

        dynamic_elbow_zero = None
        shoulder_progress = None

        calibration_valid = False

        shoulder_range_deg = None
        elbow_range_deg = None

        if (
            raw_angles is not None
            and neutral_calibration is not None
            and horizontal_calibration is not None
            and elbow_90_calibration is not None
        ):
            shoulder_range_deg = abs(degrees(
                horizontal_calibration["shoulder_lateral"]
                - neutral_calibration["shoulder_lateral"]
            ))
            elbow_range_deg = abs(degrees(
                elbow_90_calibration["elbow_flexion"]
                - horizontal_calibration["elbow_flexion"]
            ))

            calibration_valid = (
                shoulder_range_deg
                >= CONFIG["MIN_SHOULDER_CALIBRATION_RANGE_DEG"]
                and elbow_range_deg
                >= CONFIG["MIN_ELBOW_CALIBRATION_RANGE_DEG"]
            )

            if calibration_valid:
                shoulder_preview = map_calibrated_value(
                    raw_angles["shoulder_lateral"],
                    neutral_calibration["shoulder_lateral"],
                    horizontal_calibration["shoulder_lateral"],
                    np.pi / 2.0,
                )

                dynamic_elbow_zero, shoulder_progress = (
                    calculate_dynamic_elbow_zero(
                        current_shoulder=(
                            raw_angles["shoulder_lateral"]
                        ),
                        neutral_shoulder=(
                            neutral_calibration[
                                "shoulder_lateral"
                            ]
                        ),
                        horizontal_shoulder=(
                            horizontal_calibration[
                                "shoulder_lateral"
                            ]
                        ),
                        neutral_elbow=(
                            neutral_calibration[
                                "elbow_flexion"
                            ]
                        ),
                        horizontal_elbow=(
                            horizontal_calibration[
                                "elbow_flexion"
                            ]
                        ),
                    )
                )

                if dynamic_elbow_zero is not None:

                    elbow_calibration_range = (
                        elbow_90_calibration[
                            "elbow_flexion"
                        ]
                        - horizontal_calibration[
                            "elbow_flexion"
                        ]
                    )

                    if abs(elbow_calibration_range) > 1e-6:

                        elbow_normalized = (
                            raw_angles["elbow_flexion"]
                            - dynamic_elbow_zero
                        ) / elbow_calibration_range

                        elbow_normalized = float(
                            np.clip(
                                elbow_normalized,
                                0.0,
                                1.0,
                            )
                        )

                        elbow_preview = float(
                            elbow_normalized
                            * (np.pi / 2.0)
                        )

        if (
            calibration_valid
            and not calibration_was_valid
        ):
            shoulder_preview_smoothed = None
            elbow_preview_smoothed = None

        calibration_was_valid = calibration_valid

        if calibration_valid:
            shoulder_preview = clamp_angle_degrees(
                angle_radians=shoulder_preview,
                minimum_degrees=CONFIG[
                    "SHOULDER_PREVIEW_MIN_DEG"
                ],
                maximum_degrees=CONFIG[
                    "SHOULDER_PREVIEW_MAX_DEG"
                ],
            )

            elbow_preview = clamp_angle_degrees(
                angle_radians=elbow_preview,
                minimum_degrees=CONFIG[
                    "ELBOW_PREVIEW_MIN_DEG"
                ],
                maximum_degrees=CONFIG[
                    "ELBOW_PREVIEW_MAX_DEG"
                ],
            )

            shoulder_preview = apply_dead_zone(
                angle_radians=shoulder_preview,
                dead_zone_degrees=CONFIG[
                    "ARM_DEAD_ZONE_DEG"
                ],
            )

            elbow_preview = apply_dead_zone(
                angle_radians=elbow_preview,
                dead_zone_degrees=CONFIG[
                    "ARM_DEAD_ZONE_DEG"
                ],
            )

            shoulder_preview_smoothed = smooth_scalar(
                previous_value=shoulder_preview_smoothed,
                current_value=shoulder_preview,
                alpha=CONFIG[
                    "ARM_PREVIEW_SMOOTHING_ALPHA"
                ],
            )

            elbow_preview_smoothed = smooth_scalar(
                previous_value=elbow_preview_smoothed,
                current_value=elbow_preview,
                alpha=CONFIG[
                    "ARM_PREVIEW_SMOOTHING_ALPHA"
                ],
            )

            shoulder_preview_smoothed = apply_dead_zone(
                angle_radians=shoulder_preview_smoothed,
                dead_zone_degrees=CONFIG[
                    "ARM_DEAD_ZONE_DEG"
                ],
            )

            elbow_preview_smoothed = apply_dead_zone(
                angle_radians=elbow_preview_smoothed,
                dead_zone_degrees=CONFIG[
                    "ARM_DEAD_ZONE_DEG"
                ],
            )

            if (
                shoulder_preview_smoothed is not None
                and elbow_preview_smoothed is not None
            ):
                current_pose_name = classify_calibrated_pose(
                    shoulder_degrees=degrees(
                        shoulder_preview_smoothed
                    ),
                    elbow_degrees=degrees(
                        elbow_preview_smoothed
                    ),
                    tolerance_degrees=CONFIG[
                        "ARM_POSE_VALIDATION_TOLERANCE_DEG"
                    ],
                )
            else:
                current_pose_name = "SIN CALIBRAR"
        else:
            current_pose_name = "SIN CALIBRAR"

        display = cv2.flip(frame, 1)
        display_height, display_width, _ = display.shape

        put_text(display, f"FPS: {fps_display:.1f}", 35)
        put_text(
            display,
            "TORSO VALIDO" if data.get("torso_reference_valid", False)
            else "TORSO NO VALIDO",
            70,
            (0, 255, 0) if data.get("torso_reference_valid", False)
            else (0, 0, 255),
        )
        put_text(
            display,
            "BRAZO VALIDO" if data.get("arm_detected", False)
            else "BRAZO NO VALIDO",
            100,
            (0, 255, 0) if data.get("arm_detected", False)
            else (0, 0, 255),
        )

        if raw_angles is not None:
            put_text(
                display,
                f"Hombro lateral world: {degrees(raw_angles['shoulder_lateral']):.1f} deg",
                140,
                (255, 255, 0),
            )
            put_text(
                display,
                f"Hombro frente/atras world: {degrees(raw_angles['shoulder_forward']):.1f} deg",
                170,
                (255, 255, 0),
            )
            put_text(
                display,
                f"Flexion codo world: {degrees(raw_angles['elbow_flexion']):.1f} deg",
                200,
                (255, 255, 0),
            )

        put_text(
            display,
            calibration_message,
            240,
            (0, 255, 0) if calibration_valid else (0, 165, 255),
            0.50,
        )
        put_text(
            display,
            f"Neutral: {'OK' if neutral_calibration else 'PENDIENTE'}",
            275,
        )
        put_text(
            display,
            f"Horizontal: {'OK' if horizontal_calibration else 'PENDIENTE'}",
            305,
        )
        put_text(
            display,
            f"Codo 90: {'OK' if elbow_90_calibration else 'PENDIENTE'}",
            335,
        )

        if shoulder_range_deg is not None and elbow_range_deg is not None:
            put_text(
                display,
                (
                    f"Rangos | Hombro C-H: {shoulder_range_deg:.1f} deg | "
                    f"Codo H-E: {elbow_range_deg:.1f} deg"
                ),
                365,
                (255, 255, 255),
                0.50,
            )

        if (
            calibration_valid
            and shoulder_preview_smoothed is not None
            and elbow_preview_smoothed is not None
        ):
            put_text(
                display,
                (
                    "Preview hombro suavizado: "
                    f"{degrees(shoulder_preview_smoothed):.1f} deg"
                ),
                400,
                (0, 255, 0),
            )
            put_text(
                display,
                (
                    "Preview codo suavizado: "
                    f"{degrees(elbow_preview_smoothed):.1f} deg"
                ),
                430,
                (0, 255, 0),
            )
        elif shoulder_range_deg is not None:
            put_text(
                display,
                (
                    f"Calibracion invalida | Hombro: {shoulder_range_deg:.1f} | "
                    f"Codo: {elbow_range_deg:.1f}"
                ),
                400,
                (0, 0, 255),
            )

        if (
            dynamic_elbow_zero is not None
            and shoulder_progress is not None
        ):
            put_text(
                display,
                (
                    "Cero dinamico codo: "
                    f"{degrees(dynamic_elbow_zero):.1f} deg | "
                    "Posicion hombro: "
                    f"{shoulder_progress * 100.0:.1f} %"
                ),
                460,
                (255, 255, 255),
                0.50,
            )

        pose_color = (
            (0, 255, 0)
            if current_pose_name
            in (
                "C - NEUTRAL",
                "H - HORIZONTAL RECTO",
                "E - CODO 90",
            )
            else (0, 165, 255)
        )

        put_text(
            display,
            (
                "Postura calibrada: "
                f"{current_pose_name}"
            ),
            490,
            pose_color,
            0.55,
        )

        put_text(
            display,
            (
                "C neutral | H horizontal recto | "
                "E muneca arriba | X reiniciar"
            ),
            525,
            (220, 220, 220),
            0.50,
            1,
        )
        put_text(
            display,
            "ESC: salir",
            display_height - 20,
            (220, 220, 220),
            0.50,
            1,
        )

        cv2.imshow("Calibracion de brazo world", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key in (ord("c"), ord("C")):
            if raw_angles is None:
                calibration_message = "No hay brazo/torso world valido"
            else:
                calibration_mode = "neutral"
                calibration_samples = []
                calibration_message = "Manten brazo abajo y codo recto"
        elif key in (ord("h"), ord("H")):
            if raw_angles is None:
                calibration_message = "No hay brazo/torso world valido"
            else:
                calibration_mode = "horizontal"
                calibration_samples = []
                calibration_message = "Manten brazo horizontal y codo recto"
        elif key in (ord("e"), ord("E")):
            if raw_angles is None:
                calibration_message = "No hay brazo/torso world valido"
            else:
                calibration_mode = "elbow90"
                calibration_samples = []
                calibration_message = "Brazo superior horizontal, muneca arriba"
        elif key in (ord("x"), ord("X")):
            calibration_mode = None
            calibration_samples = []
            neutral_calibration = None
            horizontal_calibration = None
            elbow_90_calibration = None
            shoulder_preview_smoothed = None
            elbow_preview_smoothed = None
            current_pose_name = "SIN CALIBRAR"
            calibration_was_valid = False
            calibration_message = "Calibracion reiniciada"

except KeyboardInterrupt:
    print("Programa detenido con Ctrl + C.")
finally:
    cap.release()
    pose_detector.close()
    hand_detector.close()
    cv2.destroyAllWindows()
    print("Recursos liberados correctamente.")

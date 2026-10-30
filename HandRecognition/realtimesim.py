# Simulación en tiempo real de un robot URDF
# controlado mediante MediaPipe.

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
# RUTAS
# ============================================================

BASE_DIRECTORY = Path(__file__).parent

URDF_FILENAME = "hachi.urdf"

URDF_PATH = (
    BASE_DIRECTORY
    / URDF_FILENAME
)

POSE_MODEL = (
    BASE_DIRECTORY
    / "modelos"
    / "pose_landmarker_full.task"
)

HAND_MODEL = (
    BASE_DIRECTORY
    / "modelos"
    / "hand_landmarker.task"
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

CAMERA_INDEX = 0

# Brazo anatómico real:
# "left"  = brazo izquierdo real.
# "right" = brazo derecho real.
TARGET_ARM = "right"


CONFIG = {
    # --------------------------------------------------------
    # MediaPipe Pose
    # --------------------------------------------------------
    "POSE_DETECTION_CONFIDENCE": 0.2,
    "POSE_PRESENCE_CONFIDENCE": 0.2,
    "POSE_TRACKING_CONFIDENCE": 0.2,

    # --------------------------------------------------------
    # MediaPipe Hand
    # --------------------------------------------------------
    "HAND_DETECTION_CONFIDENCE": 0.2,
    "HAND_PRESENCE_CONFIDENCE": 0.2,
    "HAND_TRACKING_CONFIDENCE": 0.2,

    # Se conservan dos manos para seleccionar la más cercana
    # a la muñeca del brazo objetivo.
    "NUM_HANDS": 2,

    # --------------------------------------------------------
    # Cámara
    # --------------------------------------------------------
    "CAMERA_WIDTH": 1280,
    "CAMERA_HEIGHT": 720,
    "CAMERA_FPS": 30,

    # --------------------------------------------------------
    # Resolución utilizada por MediaPipe
    # --------------------------------------------------------
    "PROCESSING_WIDTH": 640,
    "PROCESSING_HEIGHT": 360,

    # --------------------------------------------------------
    # Procesamiento
    # --------------------------------------------------------
    "SMOOTHING_ALPHA": 0.35,
    "MAX_HAND_WRIST_DISTANCE": 0.30,

    # --------------------------------------------------------
    # Diagnóstico
    # --------------------------------------------------------
    "DEBUG_PRINT_COORDINATES": False,
    "PRINT_INTERVAL_SECONDS": 1.0,
    "SHOW_PROCESSING_TIME": True,

    # --------------------------------------------------------
    # Swift
    # --------------------------------------------------------
    "SWIFT_STEP_SECONDS": 0.01,

    # Cantidad de frames durante los cuales se conserva
    # la última posición válida si se pierde la detección.
    "LOST_FRAME_TOLERANCE": 5,
}


# ============================================================
# VERIFICAR ARCHIVOS
# ============================================================

if not URDF_PATH.is_file():
    raise FileNotFoundError(
        "No se encontró el archivo URDF:\n"
        f"{URDF_PATH}"
    )

if not POSE_MODEL.is_file():
    raise FileNotFoundError(
        "No se encontró el modelo Pose Landmarker:\n"
        f"{POSE_MODEL}"
    )

if not HAND_MODEL.is_file():
    raise FileNotFoundError(
        "No se encontró el modelo Hand Landmarker:\n"
        f"{HAND_MODEL}"
    )


# ============================================================
# ABRIR CÁMARA
# ============================================================

def open_camera(camera_index):
    """
    Abre la cámara utilizando un backend compatible con
    el sistema operativo actual.
    """

    operating_system = platform.system().lower()

    capture = None

    if operating_system == "windows":

        print("Sistema operativo detectado: Windows")
        print("Intentando abrir cámara mediante DirectShow...")

        capture = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW
        )

    elif operating_system == "linux":

        print("Sistema operativo detectado: Linux")
        print("Intentando abrir cámara mediante V4L2...")

        capture = cv2.VideoCapture(
            camera_index,
            cv2.CAP_V4L2
        )

    else:

        print(
            "Sistema operativo detectado: "
            f"{platform.system()}"
        )

        capture = cv2.VideoCapture(
            camera_index
        )

    # Si falla el backend específico, usar el backend
    # automático de OpenCV.
    if capture is None or not capture.isOpened():

        print(
            "El backend específico no abrió la cámara. "
            "Intentando el modo automático..."
        )

        if capture is not None:
            capture.release()

        capture = cv2.VideoCapture(
            camera_index
        )

    if not capture.isOpened():
        raise RuntimeError(
            "No se pudo abrir la cámara.\n"
            "Verifica que no esté siendo utilizada por otra "
            "aplicación y que CAMERA_INDEX sea correcto."
        )

    return capture


# ============================================================
# CARGAR ROBOT
# ============================================================

print("=" * 70)
print("Cargando robot desde el archivo URDF...")
print(f"URDF: {URDF_PATH}")
print("=" * 70)

# Esta línea puede producir un DeprecationWarning,
# pero continúa funcionando correctamente.
robot = rtb.ERobot.URDF(
    str(URDF_PATH)
)

print(robot)


# ============================================================
# INICIAR SWIFT
# ============================================================

print("=" * 70)
print("Iniciando simulador Swift...")
print("=" * 70)

env = swift.Swift()

env.launch(
    realtime=True
)

env.add(robot)


# ============================================================
# VALIDAR ROBOT
# ============================================================

if robot.n < 3:
    raise RuntimeError(
        "El robot necesita al menos tres articulaciones "
        "para controlar hombro y codo."
    )

if robot.qlim is None:
    raise RuntimeError(
        "El robot no contiene límites articulares válidos."
    )

print("Límites articulares detectados:")

for joint_index in range(robot.n):

    joint_minimum = robot.qlim[
        0,
        joint_index
    ]

    joint_maximum = robot.qlim[
        1,
        joint_index
    ]

    print(
        f"Joint {joint_index + 1}: "
        f"{joint_minimum:.4f} a "
        f"{joint_maximum:.4f} rad"
    )


# ============================================================
# INICIAR CÁMARA
# ============================================================

cap = open_camera(
    CAMERA_INDEX
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CONFIG["CAMERA_WIDTH"]
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CONFIG["CAMERA_HEIGHT"]
)

cap.set(
    cv2.CAP_PROP_FPS,
    CONFIG["CAMERA_FPS"]
)

# Reducir el buffer cuando el backend lo permita.
cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


# ============================================================
# MOSTRAR CONFIGURACIÓN REAL DE CÁMARA
# ============================================================

actual_camera_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

actual_camera_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

actual_camera_fps = cap.get(
    cv2.CAP_PROP_FPS
)

print("=" * 70)
print("Cámara iniciada correctamente")
print(
    "Resolución reportada: "
    f"{actual_camera_width} x "
    f"{actual_camera_height}"
)
print(
    "FPS reportados por la cámara: "
    f"{actual_camera_fps:.2f}"
)
print(
    "Resolución de inferencia: "
    f"{CONFIG['PROCESSING_WIDTH']} x "
    f"{CONFIG['PROCESSING_HEIGHT']}"
)
print(
    f"Brazo objetivo: "
    f"{TARGET_ARM.upper()}"
)
print("=" * 70)


# ============================================================
# INICIAR MEDIAPIPE
# ============================================================

pose_detector, hand_detector = initialize_detectors(
    POSE_MODEL,
    HAND_MODEL,
    CONFIG
)

processor = ArmHandProcessor(
    pose_detector=pose_detector,
    hand_detector=hand_detector,
    target_arm=TARGET_ARM,
    config=CONFIG,
    use_smoothing=True
)


# ============================================================
# VARIABLES DE TIEMPO
# ============================================================

program_start_time = time.perf_counter()

last_timestamp_ms = -1
last_print_time = 0.0

fps_counter = 0
fps_display = 0.0
fps_start_time = time.perf_counter()

vision_time_ms = 0.0
swift_time_ms = 0.0
total_time_ms = 0.0


# ============================================================
# VARIABLES DEL ROBOT
# ============================================================

# Posición inicial del robot.
last_valid_q = np.zeros(
    robot.n,
    dtype=np.float64
)

# Asegurar que la posición inicial respete los límites.
last_valid_q = np.clip(
    last_valid_q,
    robot.qlim[0, :],
    robot.qlim[1, :]
)

robot.q = last_valid_q.copy()

lost_frame_count = 0


print("=" * 70)
print(
    "Sistema iniciado | "
    f"Control en tiempo real de "
    f"{TARGET_ARM.upper()}"
)
print("Presiona ESC en la ventana de cámara para salir.")
print("=" * 70)


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

try:

    while True:

        frame_start_time = time.perf_counter()

        # ----------------------------------------------------
        # CAPTURAR FRAME
        # ----------------------------------------------------

        success, frame = cap.read()

        if not success or frame is None:

            print(
                "ERROR: No se pudo obtener una imagen "
                "de la cámara."
            )

            break

        # MediaPipe recibe la imagen original, no reflejada.
        frame_height, frame_width, _ = frame.shape

        # ----------------------------------------------------
        # CALCULAR FPS
        # ----------------------------------------------------

        fps_counter += 1

        current_time = time.perf_counter()

        elapsed_fps_time = (
            current_time - fps_start_time
        )

        if elapsed_fps_time >= 1.0:

            fps_display = (
                fps_counter
                / elapsed_fps_time
            )

            fps_counter = 0
            fps_start_time = current_time

        # ----------------------------------------------------
        # TIMESTAMP DE MEDIAPIPE
        # ----------------------------------------------------

        elapsed_program_time = (
            current_time
            - program_start_time
        )

        timestamp_ms = int(
            elapsed_program_time * 1000
        )

        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1

        last_timestamp_ms = timestamp_ms

        # ----------------------------------------------------
        # REDUCIR RESOLUCIÓN PARA MEDIAPIPE
        # ----------------------------------------------------

        processing_frame = cv2.resize(
            frame,
            (
                CONFIG["PROCESSING_WIDTH"],
                CONFIG["PROCESSING_HEIGHT"]
            ),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # CONVERTIR BGR A RGB
        # ----------------------------------------------------

        rgb_processing_frame = cv2.cvtColor(
            processing_frame,
            cv2.COLOR_BGR2RGB
        )

        rgb_processing_frame = np.ascontiguousarray(
            rgb_processing_frame
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_processing_frame
        )

        # ----------------------------------------------------
        # PROCESAR MEDIAPIPE
        # ----------------------------------------------------

        vision_start_time = time.perf_counter()

        data = processor.process_frame(
            mp_image=mp_image,
            frame=frame,
            timestamp_ms=timestamp_ms
        )

        vision_end_time = time.perf_counter()

        vision_time_ms = (
            vision_end_time
            - vision_start_time
        ) * 1000.0

        # ----------------------------------------------------
        # CALCULAR POSICIONES ARTICULARES
        # ----------------------------------------------------

        tracking_valid = False

        if (
            data["arm_detected"]
            and data["shoulder"] is not None
            and data["elbow"] is not None
            and data["wrist"] is not None
        ):

            shoulder = np.asarray(
                data["shoulder"],
                dtype=np.float64
            )

            elbow = np.asarray(
                data["elbow"],
                dtype=np.float64
            )

            wrist = np.asarray(
                data["wrist"],
                dtype=np.float64
            )

            # Vector del brazo superior:
            # hombro -> codo.
            upper_arm_vector = (
                elbow - shoulder
            )

            # Vector del antebrazo:
            # codo -> muñeca.
            forearm_vector = (
                wrist - elbow
            )

            upper_arm_norm = np.linalg.norm(
                upper_arm_vector
            )

            forearm_norm = np.linalg.norm(
                forearm_vector
            )

            # Evitar cálculos inválidos con vectores
            # demasiado pequeños.
            if (
                upper_arm_norm > 1e-6
                and forearm_norm > 1e-6
            ):

                q_target = last_valid_q.copy()

                # --------------------------------------------
                # JOINT 1: SHOULDER YAW
                # --------------------------------------------

                if robot.n >= 1:

                    q_target[0] = np.arctan2(
                        upper_arm_vector[0],
                        -upper_arm_vector[2]
                    )

                # --------------------------------------------
                # JOINT 2: SHOULDER PITCH
                # --------------------------------------------

                if robot.n >= 2:

                    horizontal_distance = np.linalg.norm(
                        [
                            upper_arm_vector[0],
                            upper_arm_vector[2]
                        ]
                    )

                    q_target[1] = np.arctan2(
                        upper_arm_vector[1],
                        horizontal_distance
                    )

                # --------------------------------------------
                # JOINT 3: ELBOW PITCH
                # --------------------------------------------

                if robot.n >= 3:

                    cosine_angle = np.dot(
                        upper_arm_vector,
                        forearm_vector
                    ) / (
                        upper_arm_norm
                        * forearm_norm
                    )

                    cosine_angle = np.clip(
                        cosine_angle,
                        -1.0,
                        1.0
                    )

                    q_target[2] = np.arccos(
                        cosine_angle
                    )

                # --------------------------------------------
                # JOINTS 4 Y 5
                # --------------------------------------------
                # Todavía no se calculan la rotación y el pitch
                # de muñeca. Por ahora conservan su posición
                # anterior para evitar regresarlos a cero.

                # --------------------------------------------
                # JOINT 6: GRIPPER
                # --------------------------------------------
                # Todavía utiliza un valor provisional.
                # Más adelante se calculará mediante los
                # landmarks del pulgar, índice y dedo medio.

                if (
                    data["hand_detected"]
                    and robot.n >= 6
                ):
                    q_target[5] = 0.5

                # Respetar límites definidos por el URDF.
                q_target = np.clip(
                    q_target,
                    robot.qlim[0, :],
                    robot.qlim[1, :]
                )

                last_valid_q = q_target.copy()

                lost_frame_count = 0
                tracking_valid = True

        # ----------------------------------------------------
        # MANEJAR PÉRDIDA TEMPORAL DE DETECCIÓN
        # ----------------------------------------------------

        if not tracking_valid:

            lost_frame_count += 1

            # Durante los primeros frames perdidos se conserva
            # la última posición válida.
            if (
                lost_frame_count
                <= CONFIG["LOST_FRAME_TOLERANCE"]
            ):
                q_target = last_valid_q.copy()

            else:
                # Por seguridad, después de superar la
                # tolerancia también se mantiene la última
                # posición, sin generar nuevos movimientos.
                q_target = last_valid_q.copy()

        # ----------------------------------------------------
        # ACTUALIZAR SWIFT
        # ----------------------------------------------------

        swift_start_time = time.perf_counter()

        robot.q = q_target

        env.step(
            CONFIG["SWIFT_STEP_SECONDS"]
        )

        swift_end_time = time.perf_counter()

        swift_time_ms = (
            swift_end_time
            - swift_start_time
        ) * 1000.0

        # ----------------------------------------------------
        # IMPRESIÓN OPCIONAL DE COORDENADAS
        # ----------------------------------------------------

        if (
            CONFIG["DEBUG_PRINT_COORDINATES"]
            and data["arm_detected"]
            and data["hand_detected"]
            and (
                current_time - last_print_time
                >= CONFIG["PRINT_INTERVAL_SECONDS"]
            )
        ):

            print_robot_coordinates(
                timestamp_ms=timestamp_ms,
                target_arm=TARGET_ARM,
                shoulder=data["shoulder"],
                elbow=data["elbow"],
                wrist=data["wrist"],
                hand=data["hand"]
            )

            last_print_time = current_time

        # ----------------------------------------------------
        # CALCULAR TIEMPO TOTAL DEL FRAME
        # ----------------------------------------------------

        frame_end_time = time.perf_counter()

        total_time_ms = (
            frame_end_time
            - frame_start_time
        ) * 1000.0

        # ----------------------------------------------------
        # CREAR VISTA ESPEJO
        # ----------------------------------------------------

        # Los landmarks ya fueron dibujados sobre el frame.
        # Al reflejarlo ahora, la cámara y los landmarks
        # permanecen alineados.
        display_frame = cv2.flip(
            frame,
            1
        )

        display_height, display_width, _ = (
            display_frame.shape
        )

        # ----------------------------------------------------
        # INFORMACIÓN EN PANTALLA
        # ----------------------------------------------------
        # Los textos se dibujan después del flip para que no
        # aparezcan reflejados.

        cv2.putText(
            display_frame,
            f"Brazo: {TARGET_ARM.upper()}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"FPS: {fps_display:.1f}",
            (display_width - 170, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2
        )

        arm_status_text = (
            "BRAZO DETECTADO"
            if data["arm_detected"]
            else "BRAZO NO DETECTADO"
        )

        arm_status_color = (
            (0, 255, 0)
            if data["arm_detected"]
            else (0, 0, 255)
        )

        cv2.putText(
            display_frame,
            arm_status_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            arm_status_color,
            2
        )

        hand_status_text = (
            "MANO DETECTADA"
            if data["hand_detected"]
            else "MANO NO DETECTADA"
        )

        hand_status_color = (
            (0, 255, 0)
            if data["hand_detected"]
            else (0, 0, 255)
        )

        cv2.putText(
            display_frame,
            hand_status_text,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            hand_status_color,
            2
        )

        cv2.putText(
            display_frame,
            (
                "Inferencia: "
                f"{CONFIG['PROCESSING_WIDTH']}x"
                f"{CONFIG['PROCESSING_HEIGHT']}"
            ),
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        if CONFIG["SHOW_PROCESSING_TIME"]:

            cv2.putText(
                display_frame,
                f"Vision: {vision_time_ms:.1f} ms",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"Swift: {swift_time_ms:.1f} ms",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"Total: {total_time_ms:.1f} ms",
                (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        tracking_status = (
            "TRACKING ACTIVO"
            if tracking_valid
            else (
                "MANTENIENDO POSICION | "
                f"Frames perdidos: {lost_frame_count}"
            )
        )

        tracking_color = (
            (0, 255, 0)
            if tracking_valid
            else (0, 165, 255)
        )

        cv2.putText(
            display_frame,
            tracking_status,
            (20, 265),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            tracking_color,
            2
        )

        cv2.putText(
            display_frame,
            "ESC: salir",
            (20, display_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1
        )

        # ----------------------------------------------------
        # MOSTRAR CÁMARA
        # ----------------------------------------------------

        cv2.imshow(
            "MediaPipe - Control de robot",
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:

            print(
                "\nPrograma detenido por el usuario."
            )

            break


# ============================================================
# LIBERAR RECURSOS
# ============================================================

except KeyboardInterrupt:

    print(
        "\nPrograma detenido mediante Ctrl + C."
    )

finally:

    print("Liberando recursos...")

    cap.release()

    pose_detector.close()
    hand_detector.close()

    cv2.destroyAllWindows()

    try:
        env.close()
    except Exception as close_error:
        print(
            "Swift no pudo cerrarse automáticamente: "
            f"{close_error}"
        )

    print("Recursos liberados correctamente.")
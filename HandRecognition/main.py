from pathlib import Path
import platform
import time

import cv2
import mediapipe as mp
import numpy as np

from detector import initialize_detectors
from processor import ArmHandProcessor
from utils import print_robot_coordinates


# ============================================================
# RUTAS DE MODELOS
# ============================================================

BASE_DIRECTORY = Path(__file__).parent

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

# Brazo anatómico real de la persona:
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

    # Se conservan dos manos porque, si ambas aparecen,
    # el programa necesita elegir la más cercana a la muñeca
    # del brazo controlado.
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
    # La cámara continúa mostrando 1280x720, pero la inferencia
    # se realiza en 640x360 para reducir procesamiento.
    "PROCESSING_WIDTH": 640,
    "PROCESSING_HEIGHT": 360,

    # --------------------------------------------------------
    # Procesamiento y suavizado
    # --------------------------------------------------------
    "SMOOTHING_ALPHA": 0.35,
    "MAX_HAND_WRIST_DISTANCE": 0.30,
        # --------------------------------------------------------
    # Validación y tolerancia
    # --------------------------------------------------------
    "LANDMARK_VISIBILITY_THRESHOLD": 0.20,
    "ARM_LOST_FRAME_TOLERANCE": 5,
    "HAND_LOST_FRAME_TOLERANCE": 5,

    # --------------------------------------------------------
    # Visualización
    # --------------------------------------------------------
    "DRAW_ARM": True,
    "DRAW_HAND": True,

    # True muestra los 21 landmarks.
    # False muestra únicamente muñeca, pulgar, índice y medio.
    "DRAW_FULL_HAND": False,

    # --------------------------------------------------------
    # Diagnóstico
    # --------------------------------------------------------
    "DEBUG_PRINT_COORDINATES": False,
    "PRINT_INTERVAL_SECONDS": 1.0,

    # Mostrar el tiempo aproximado dedicado a procesar
    # Pose Landmarker, Hand Landmarker y el procesador.
    "SHOW_PROCESSING_TIME": True,
}


# ============================================================
# FUNCIÓN PARA ABRIR LA CÁMARA
# ============================================================

def open_camera(camera_index):
    """
    Abre la cámara utilizando un backend apropiado según
    el sistema operativo.

    Windows:
        Intenta primero DirectShow y después modo automático.

    Linux:
        Intenta primero Video4Linux2 y después modo automático.

    Otros sistemas:
        Utiliza selección automática de OpenCV.
    """

    operating_system = platform.system().lower()

    capture = None

    if operating_system == "windows":
        print("Sistema operativo detectado: Windows")
        print("Intentando abrir cámara con DirectShow...")

        capture = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW
        )

    elif operating_system == "linux":
        print("Sistema operativo detectado: Linux")
        print("Intentando abrir cámara con V4L2...")

        capture = cv2.VideoCapture(
            camera_index,
            cv2.CAP_V4L2
        )

    else:
        print(
            "Sistema operativo detectado: "
            f"{platform.system()}"
        )

        capture = cv2.VideoCapture(camera_index)

    # Si el backend específico falla, OpenCV intenta
    # seleccionar automáticamente otro backend.
    if capture is None or not capture.isOpened():
        print(
            "El backend específico no pudo abrir la cámara. "
            "Intentando modo automático..."
        )

        if capture is not None:
            capture.release()

        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        raise RuntimeError(
            "No se pudo abrir la cámara.\n"
            "Verifica que no esté siendo utilizada por otra "
            "aplicación y que CAMERA_INDEX sea correcto."
        )

    return capture


# ============================================================
# INICIAR CÁMARA
# ============================================================

cap = open_camera(CAMERA_INDEX)

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
# Esto ayuda a evitar que se acumulen frames retrasados.
cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


# ============================================================
# MOSTRAR CONFIGURACIÓN REAL DE LA CÁMARA
# ============================================================

actual_camera_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

actual_camera_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

actual_camera_fps = cap.get(
    cv2.CAP_PROP_FPS
)

print("=" * 70)
print("Cámara iniciada correctamente")
print(
    "Resolución reportada por la cámara: "
    f"{actual_camera_width} x {actual_camera_height}"
)
print(
    "FPS reportados por la cámara: "
    f"{actual_camera_fps:.2f}"
)
print(
    "Resolución de procesamiento: "
    f"{CONFIG['PROCESSING_WIDTH']} x "
    f"{CONFIG['PROCESSING_HEIGHT']}"
)
print(f"Brazo anatómico objetivo: {TARGET_ARM.upper()}")
print(f"Pose model: {POSE_MODEL}")
print(f"Hand model: {HAND_MODEL}")
print("=" * 70)


# ============================================================
# INICIALIZAR MEDIAPIPE
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

processing_time_ms = 0.0


print("=" * 70)
print(
    "Sistema iniciado | "
    f"Brazo objetivo: {TARGET_ARM.upper()}"
)
print("Presiona ESC en la ventana de cámara para salir.")
print("=" * 70)


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

try:

    while True:

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

        # La imagen NO se refleja antes de MediaPipe.
        # Así se conserva la lateralidad anatómica real:
        # "right" representa el brazo derecho real.
        frame_height, frame_width, _ = frame.shape

        # ----------------------------------------------------
        # CALCULAR FPS
        # ----------------------------------------------------

        fps_counter += 1

        current_time = time.perf_counter()

        fps_elapsed_time = (
            current_time - fps_start_time
        )

        if fps_elapsed_time >= 1.0:

            fps_display = (
                fps_counter / fps_elapsed_time
            )

            fps_counter = 0
            fps_start_time = current_time

        # ----------------------------------------------------
        # TIMESTAMP PARA MEDIAPIPE
        # ----------------------------------------------------

        elapsed_seconds = (
            current_time - program_start_time
        )

        timestamp_ms = int(
            elapsed_seconds * 1000
        )

        # MediaPipe en modo VIDEO requiere timestamps
        # estrictamente crecientes.
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
        # PROCESAR POSE Y MANO
        # ----------------------------------------------------

        processing_start_time = time.perf_counter()

        # Los landmarks están normalizados. Por eso pueden
        # detectarse en 640x360 y dibujarse sobre 1280x720.
        data = processor.process_frame(
            mp_image=mp_image,
            frame=frame,
            timestamp_ms=timestamp_ms
        )

        processing_end_time = time.perf_counter()

        processing_time_ms = (
            processing_end_time
            - processing_start_time
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
        # CREAR VISTA ESPEJO
        # ----------------------------------------------------

        # El processor dibujó los landmarks sobre la imagen
        # original. Al reflejarla, tanto la cámara como los
        # landmarks quedan alineados en modo espejo.
        display_frame = cv2.flip(frame, 1)

        display_height, display_width, _ = (
            display_frame.shape
        )

        # ----------------------------------------------------
        # INFORMACIÓN GENERAL
        # ----------------------------------------------------
        # El texto se dibuja DESPUÉS de reflejar la imagen,
        # de modo que permanece normal y legible.

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

        status_arm_text = (
            "BRAZO DETECTADO"
            if data["arm_detected"]
            else "BRAZO NO DETECTADO"
        )

        status_arm_color = (
            (0, 255, 0)
            if data["arm_detected"]
            else (0, 0, 255)
        )

        cv2.putText(
            display_frame,
            status_arm_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            status_arm_color,
            2
        )

        status_hand_text = (
            "MANO DETECTADA"
            if data["hand_detected"]
            else "MANO NO DETECTADA"
        )

        status_hand_color = (
            (0, 255, 0)
            if data["hand_detected"]
            else (0, 0, 255)
        )

        cv2.putText(
            display_frame,
            status_hand_text,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_hand_color,
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
                (
                    "Procesamiento: "
                    f"{processing_time_ms:.1f} ms"
                ),
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
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
        # MOSTRAR IMAGEN
        # ----------------------------------------------------

        cv2.imshow(
            "MediaPipe - Control de robot",
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("\nPrograma detenido por el usuario.")
            break


# ============================================================
# LIBERAR RECURSOS
# ============================================================

finally:

    cap.release()

    pose_detector.close()
    hand_detector.close()

    cv2.destroyAllWindows()

    print("Recursos liberados correctamente.")
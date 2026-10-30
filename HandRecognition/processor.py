import time

import cv2
import numpy as np

from utils import (
    HAND_CONNECTIONS,
    draw_line,
    draw_point,
    landmark_is_visible,
    normalized_landmark_to_array,
    select_hand_near_pose_wrist,
    smooth_point,
)


# ============================================================
# LANDMARKS UTILIZADOS PARA CONTROLAR LA GARRA
# ============================================================

HAND_WRIST = 0

THUMB_MCP = 2
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_TIP = 12


HAND_CONTROL_INDICES = (
    HAND_WRIST,
    THUMB_MCP,
    THUMB_TIP,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
)


# Conexiones simplificadas para mostrar solamente pulgar,
# índice, dedo medio y la referencia de la palma.
HAND_CONTROL_CONNECTIONS = (
    (HAND_WRIST, THUMB_MCP),
    (THUMB_MCP, THUMB_TIP),

    (HAND_WRIST, INDEX_MCP),
    (INDEX_MCP, INDEX_TIP),

    (HAND_WRIST, MIDDLE_MCP),
    (MIDDLE_MCP, MIDDLE_TIP),

    (INDEX_MCP, MIDDLE_MCP),
)


# ============================================================
# FUNCIONES AUXILIARES DE LA MANO
# ============================================================

def extract_hand_control_points(hand_landmarks):
    """
    Extrae solamente los landmarks necesarios para calcular
    la apertura de la garra.

    Devuelve un diccionario con índices originales de MediaPipe:

        {
            0:  ndarray([x, y, z]),
            2:  ndarray([x, y, z]),
            4:  ndarray([x, y, z]),
            5:  ndarray([x, y, z]),
            8:  ndarray([x, y, z]),
            9:  ndarray([x, y, z]),
            12: ndarray([x, y, z])
        }
    """

    return {
        landmark_index: normalized_landmark_to_array(
            hand_landmarks[landmark_index]
        )
        for landmark_index in HAND_CONTROL_INDICES
    }


def smooth_hand_control_points(
    previous_points,
    current_points,
    alpha
):
    """
    Aplica suavizado exponencial únicamente a los landmarks
    utilizados para controlar la garra.
    """

    if previous_points is None:
        return {
            index: point.copy()
            for index, point in current_points.items()
        }

    smoothed_points = {}

    for landmark_index, current_point in current_points.items():

        previous_point = previous_points.get(
            landmark_index
        )

        smoothed_points[landmark_index] = smooth_point(
            previous_point,
            current_point,
            alpha
        )

    return smoothed_points


def calculate_raw_hand_openness(control_points):
    """
    Calcula una medida cruda y continua de apertura de la mano.

    La medida utiliza:

    - Separación pulgar-índice.
    - Extensión del índice.
    - Extensión del dedo medio.

    Las distancias se normalizan utilizando el tamaño aproximado
    de la palma para reducir el efecto de acercar o alejar la mano
    de la cámara.

    El resultado todavía NO está limitado entre 0.0 y 1.0.

    Este valor se utilizará posteriormente para calibrar:

        mano cerrada -> 0.0
        mano abierta -> 1.0
    """

    wrist = control_points[HAND_WRIST]

    thumb_tip = control_points[THUMB_TIP]

    index_mcp = control_points[INDEX_MCP]
    index_tip = control_points[INDEX_TIP]

    middle_mcp = control_points[MIDDLE_MCP]
    middle_tip = control_points[MIDDLE_TIP]

    # Tamaño de referencia de la palma.
    palm_size = np.linalg.norm(
        middle_mcp - wrist
    )

    if palm_size <= 1e-6:
        return None

    # Distancia entre pulgar e índice.
    thumb_index_distance = (
        np.linalg.norm(
            thumb_tip - index_tip
        )
        / palm_size
    )

    # Extensión aproximada del índice.
    index_extension = (
        np.linalg.norm(
            index_tip - index_mcp
        )
        / palm_size
    )

    # Extensión aproximada del dedo medio.
    middle_extension = (
        np.linalg.norm(
            middle_tip - middle_mcp
        )
        / palm_size
    )

    # Se conserva una salida continua.
    # Los pesos podrán ajustarse después de la calibración.
    raw_openness = (
        0.40 * thumb_index_distance
        + 0.30 * index_extension
        + 0.30 * middle_extension
    )

    return float(raw_openness)


# ============================================================
# PROCESADOR PRINCIPAL
# ============================================================

class ArmHandProcessor:

    def __init__(
        self,
        pose_detector,
        hand_detector,
        target_arm,
        config,
        use_smoothing=True
    ):
        """
        Procesa el brazo seleccionado y la mano correspondiente.

        El procesador:

        - Ejecuta Pose Landmarker.
        - Ejecuta Hand Landmarker.
        - Selecciona la mano más cercana a la muñeca objetivo.
        - Tolera pérdidas breves de detección.
        - Suaviza los landmarks necesarios.
        - Calcula una medida cruda de apertura de mano.
        - Dibuja resultados de acuerdo con la configuración.
        """

        self.pose_detector = pose_detector
        self.hand_detector = hand_detector

        self.target_arm = target_arm.lower()

        self.use_smoothing = use_smoothing

        # ----------------------------------------------------
        # CONFIGURACIÓN DE SUAVIZADO
        # ----------------------------------------------------

        self.smoothing_alpha = config.get(
            "SMOOTHING_ALPHA",
            0.35
        )

        # ----------------------------------------------------
        # CONFIGURACIÓN DE VALIDACIÓN
        # ----------------------------------------------------

        self.max_hand_distance = config.get(
            "MAX_HAND_WRIST_DISTANCE",
            0.30
        )

        self.visibility_threshold = config.get(
            "LANDMARK_VISIBILITY_THRESHOLD",
            0.20
        )

        # ----------------------------------------------------
        # TOLERANCIA DE DETECCIÓN
        # ----------------------------------------------------

        self.arm_lost_frame_tolerance = config.get(
            "ARM_LOST_FRAME_TOLERANCE",
            config.get(
                "LOST_FRAME_TOLERANCE",
                5
            )
        )

        self.hand_lost_frame_tolerance = config.get(
            "HAND_LOST_FRAME_TOLERANCE",
            config.get(
                "LOST_FRAME_TOLERANCE",
                5
            )
        )

        # ----------------------------------------------------
        # CONFIGURACIÓN DE DIBUJO
        # ----------------------------------------------------

        self.draw_arm = config.get(
            "DRAW_ARM",
            True
        )

        self.draw_hand = config.get(
            "DRAW_HAND",
            True
        )

        # True:
        # Dibuja los 21 landmarks de la mano.
        #
        # False:
        # Dibuja solamente los landmarks utilizados para
        # controlar la garra.
        self.draw_full_hand = config.get(
            "DRAW_FULL_HAND",
            False
        )

        # ----------------------------------------------------
        # ÍNDICES DEL BRAZO
        # ----------------------------------------------------

        if self.target_arm == "left":

            self.shoulder_idx = 11
            self.elbow_idx = 13
            self.wrist_idx = 15

        elif self.target_arm == "right":

            self.shoulder_idx = 12
            self.elbow_idx = 14
            self.wrist_idx = 16

        else:

            raise ValueError(
                'TARGET_ARM debe ser "left" o "right".'
            )

        self.hand_wrist_idx = HAND_WRIST

        # ----------------------------------------------------
        # ESTADOS ANTERIORES DEL BRAZO
        # ----------------------------------------------------

        self.prev_shoulder = None
        self.prev_elbow = None
        self.prev_wrist = None

        self.arm_lost_frames = 0

        # ----------------------------------------------------
        # ESTADOS ANTERIORES DE LA MANO
        # ----------------------------------------------------

        self.prev_hand_control = None

        # Se conservan los 21 landmarks sin suavizar únicamente
        # para dibujo completo y diagnóstico.
        self.last_full_hand = None

        self.last_hand_label = "Unknown"
        self.last_hand_distance = None

        self.last_raw_hand_openness = None

        self.hand_lost_frames = 0

        # ----------------------------------------------------
        # TIEMPOS
        # ----------------------------------------------------

        self.pose_time_ms = 0.0
        self.hand_time_ms = 0.0
        self.postprocess_time_ms = 0.0

    # ========================================================
    # REINICIAR ESTADOS
    # ========================================================

    def reset_smoothing(self):
        """
        Elimina todos los estados anteriores del procesador.
        """

        self.prev_shoulder = None
        self.prev_elbow = None
        self.prev_wrist = None

        self.prev_hand_control = None
        self.last_full_hand = None

        self.last_hand_label = "Unknown"
        self.last_hand_distance = None

        self.last_raw_hand_openness = None

        self.arm_lost_frames = 0
        self.hand_lost_frames = 0

    # ========================================================
    # PROCESAR FRAME
    # ========================================================

    def process_frame(
        self,
        mp_image,
        frame,
        timestamp_ms
    ):
        """
        Procesa un frame y devuelve los datos del brazo,
        la mano, apertura cruda y tiempos de ejecución.
        """

        height, width, _ = frame.shape

        # ----------------------------------------------------
        # EJECUTAR POSE LANDMARKER
        # ----------------------------------------------------

        pose_start_time = time.perf_counter()

        pose_result = self.pose_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        pose_end_time = time.perf_counter()

        self.pose_time_ms = (
            pose_end_time
            - pose_start_time
        ) * 1000.0

        # ----------------------------------------------------
        # EJECUTAR HAND LANDMARKER
        # ----------------------------------------------------

        hand_start_time = time.perf_counter()

        hand_result = self.hand_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        hand_end_time = time.perf_counter()

        self.hand_time_ms = (
            hand_end_time
            - hand_start_time
        ) * 1000.0

        postprocess_start_time = time.perf_counter()

        # ----------------------------------------------------
        # VARIABLES DEL FRAME ACTUAL
        # ----------------------------------------------------

        arm_detected_current = False
        hand_detected_current = False

        shoulder_coords = None
        elbow_coords = None
        wrist_coords = None

        full_hand_coords = None
        hand_control_points = None

        hand_label = "Unknown"
        hand_distance = None

        raw_hand_openness = None

        # ----------------------------------------------------
        # PROCESAR BRAZO ACTUAL
        # ----------------------------------------------------

        if pose_result.pose_landmarks:

            pose_landmarks = (
                pose_result.pose_landmarks[0]
            )

            shoulder_landmark = pose_landmarks[
                self.shoulder_idx
            ]

            elbow_landmark = pose_landmarks[
                self.elbow_idx
            ]

            wrist_landmark = pose_landmarks[
                self.wrist_idx
            ]

            shoulder_visible = landmark_is_visible(
                shoulder_landmark,
                self.visibility_threshold
            )

            elbow_visible = landmark_is_visible(
                elbow_landmark,
                self.visibility_threshold
            )

            wrist_visible = landmark_is_visible(
                wrist_landmark,
                self.visibility_threshold
            )

            if (
                shoulder_visible
                and elbow_visible
                and wrist_visible
            ):

                arm_detected_current = True

                shoulder_coords = (
                    normalized_landmark_to_array(
                        shoulder_landmark
                    )
                )

                elbow_coords = (
                    normalized_landmark_to_array(
                        elbow_landmark
                    )
                )

                wrist_coords = (
                    normalized_landmark_to_array(
                        wrist_landmark
                    )
                )

                # --------------------------------------------
                # SUAVIZAR BRAZO
                # --------------------------------------------

                if self.use_smoothing:

                    shoulder_coords = smooth_point(
                        self.prev_shoulder,
                        shoulder_coords,
                        self.smoothing_alpha
                    )

                    elbow_coords = smooth_point(
                        self.prev_elbow,
                        elbow_coords,
                        self.smoothing_alpha
                    )

                    wrist_coords = smooth_point(
                        self.prev_wrist,
                        wrist_coords,
                        self.smoothing_alpha
                    )

                self.prev_shoulder = (
                    shoulder_coords.copy()
                )

                self.prev_elbow = (
                    elbow_coords.copy()
                )

                self.prev_wrist = (
                    wrist_coords.copy()
                )

                self.arm_lost_frames = 0

                # --------------------------------------------
                # SELECCIONAR MANO CORRESPONDIENTE
                # --------------------------------------------

                (
                    selected_hand,
                    hand_label,
                    hand_distance
                ) = select_hand_near_pose_wrist(
                    hand_result,
                    wrist_landmark,
                    self.max_hand_distance,
                    self.hand_wrist_idx
                )

                if selected_hand is not None:

                    hand_detected_current = True

                    # Guardar los 21 landmarks únicamente para
                    # compatibilidad, diagnóstico y dibujo
                    # completo opcional.
                    full_hand_coords = np.asarray(
                        [
                            normalized_landmark_to_array(
                                landmark
                            )
                            for landmark in selected_hand
                        ],
                        dtype=np.float32
                    )

                    # Extraer solo los puntos utilizados para
                    # controlar la garra.
                    hand_control_points = (
                        extract_hand_control_points(
                            selected_hand
                        )
                    )

                    if self.use_smoothing:

                        hand_control_points = (
                            smooth_hand_control_points(
                                self.prev_hand_control,
                                hand_control_points,
                                self.smoothing_alpha
                            )
                        )

                    raw_hand_openness = (
                        calculate_raw_hand_openness(
                            hand_control_points
                        )
                    )

                    # Guardar estados válidos.
                    self.prev_hand_control = {
                        index: point.copy()
                        for index, point
                        in hand_control_points.items()
                    }

                    self.last_full_hand = (
                        full_hand_coords.copy()
                    )

                    self.last_hand_label = hand_label
                    self.last_hand_distance = hand_distance

                    self.last_raw_hand_openness = (
                        raw_hand_openness
                    )

                    self.hand_lost_frames = 0

        # ----------------------------------------------------
        # TOLERANCIA DE PÉRDIDA DEL BRAZO
        # ----------------------------------------------------

        if not arm_detected_current:

            self.arm_lost_frames += 1

            if (
                self.arm_lost_frames
                <= self.arm_lost_frame_tolerance
                and self.prev_shoulder is not None
                and self.prev_elbow is not None
                and self.prev_wrist is not None
            ):

                shoulder_coords = (
                    self.prev_shoulder.copy()
                )

                elbow_coords = (
                    self.prev_elbow.copy()
                )

                wrist_coords = (
                    self.prev_wrist.copy()
                )

            else:

                shoulder_coords = None
                elbow_coords = None
                wrist_coords = None

        # ----------------------------------------------------
        # TOLERANCIA DE PÉRDIDA DE LA MANO
        # ----------------------------------------------------

        if not hand_detected_current:

            self.hand_lost_frames += 1

            if (
                self.hand_lost_frames
                <= self.hand_lost_frame_tolerance
                and self.prev_hand_control is not None
            ):

                hand_control_points = {
                    index: point.copy()
                    for index, point
                    in self.prev_hand_control.items()
                }

                if self.last_full_hand is not None:
                    full_hand_coords = (
                        self.last_full_hand.copy()
                    )

                hand_label = self.last_hand_label
                hand_distance = self.last_hand_distance

                raw_hand_openness = (
                    self.last_raw_hand_openness
                )

            else:

                hand_control_points = None
                full_hand_coords = None

                hand_label = "Unknown"
                hand_distance = None

                raw_hand_openness = None

        # ----------------------------------------------------
        # ESTADOS DE SEGUIMIENTO
        # ----------------------------------------------------

        arm_tracking_valid = (
            shoulder_coords is not None
            and elbow_coords is not None
            and wrist_coords is not None
        )

        hand_tracking_valid = (
            hand_control_points is not None
        )

        arm_using_previous = (
            arm_tracking_valid
            and not arm_detected_current
        )

        hand_using_previous = (
            hand_tracking_valid
            and not hand_detected_current
        )

        # ----------------------------------------------------
        # DIBUJAR BRAZO
        # ----------------------------------------------------

        if (
            self.draw_arm
            and arm_tracking_valid
        ):

            arm_color = (
                (0, 165, 255)
                if arm_using_previous
                else (0, 255, 0)
            )

            draw_line(
                frame,
                shoulder_coords,
                elbow_coords,
                width,
                height,
                arm_color,
                thickness=4
            )

            draw_line(
                frame,
                elbow_coords,
                wrist_coords,
                width,
                height,
                arm_color,
                thickness=4
            )

            draw_point(
                frame,
                shoulder_coords,
                width,
                height,
                (255, 0, 0),
                radius=8
            )

            draw_point(
                frame,
                elbow_coords,
                width,
                height,
                (0, 165, 255),
                radius=8
            )

            draw_point(
                frame,
                wrist_coords,
                width,
                height,
                (0, 0, 255),
                radius=8
            )

        # ----------------------------------------------------
        # DIBUJAR MANO
        # ----------------------------------------------------

        if (
            self.draw_hand
            and hand_tracking_valid
        ):

            hand_color = (
                (0, 165, 255)
                if hand_using_previous
                else (255, 0, 255)
            )

            # --------------------------------------------
            # DIBUJAR LOS 21 LANDMARKS
            # --------------------------------------------

            if (
                self.draw_full_hand
                and full_hand_coords is not None
            ):

                for (
                    start_index,
                    end_index
                ) in HAND_CONNECTIONS:

                    draw_line(
                        frame,
                        full_hand_coords[start_index],
                        full_hand_coords[end_index],
                        width,
                        height,
                        hand_color,
                        thickness=2
                    )

                for (
                    landmark_index,
                    landmark
                ) in enumerate(full_hand_coords):

                    point_color = (
                        (0, 0, 255)
                        if landmark_index == HAND_WRIST
                        else (0, 255, 255)
                    )

                    draw_point(
                        frame,
                        landmark,
                        width,
                        height,
                        point_color,
                        radius=4
                    )

            # --------------------------------------------
            # DIBUJO SIMPLIFICADO
            # --------------------------------------------

            else:

                for (
                    start_index,
                    end_index
                ) in HAND_CONTROL_CONNECTIONS:

                    draw_line(
                        frame,
                        hand_control_points[start_index],
                        hand_control_points[end_index],
                        width,
                        height,
                        hand_color,
                        thickness=2
                    )

                for (
                    landmark_index,
                    landmark
                ) in hand_control_points.items():

                    point_color = (
                        (0, 0, 255)
                        if landmark_index == HAND_WRIST
                        else (0, 255, 255)
                    )

                    draw_point(
                        frame,
                        landmark,
                        width,
                        height,
                        point_color,
                        radius=5
                    )

        postprocess_end_time = time.perf_counter()

        self.postprocess_time_ms = (
            postprocess_end_time
            - postprocess_start_time
        ) * 1000.0

        # ----------------------------------------------------
        # DEVOLVER RESULTADOS
        # ----------------------------------------------------

        return {
            # Compatibilidad con main.py y realtimesim.py.
            # arm_detected y hand_detected incluyen la tolerancia
            # de frames perdidos.
            "arm_detected": arm_tracking_valid,
            "hand_detected": hand_tracking_valid,

            # Detección real del frame actual.
            "arm_detected_current": arm_detected_current,
            "hand_detected_current": hand_detected_current,

            # Indica si se está reutilizando el último resultado.
            "arm_using_previous": arm_using_previous,
            "hand_using_previous": hand_using_previous,

            # Puntos del brazo.
            "shoulder": shoulder_coords,
            "elbow": elbow_coords,
            "wrist": wrist_coords,

            # Los 21 puntos quedan disponibles para compatibilidad
            # y diagnóstico.
            "hand": full_hand_coords,

            # Puntos optimizados para controlar la garra.
            "hand_control": hand_control_points,

            # Apertura continua sin calibrar.
            "raw_hand_openness": raw_hand_openness,

            # Información de asociación de la mano.
            "hand_label": hand_label,
            "distance": hand_distance,

            # Frames perdidos.
            "arm_lost_frames": self.arm_lost_frames,
            "hand_lost_frames": self.hand_lost_frames,

            # Tiempos separados.
            "pose_time_ms": self.pose_time_ms,
            "hand_time_ms": self.hand_time_ms,
            "postprocess_time_ms": self.postprocess_time_ms,
        }
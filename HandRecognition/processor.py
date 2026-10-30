import time

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
# INDICES DE POSE LANDMARKER
# ============================================================

POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24


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
# FUNCIONES AUXILIARES
# ============================================================

def normalize_vector(vector, minimum_norm=1e-6):
    """Normaliza un vector 3D o devuelve None si es degenerado."""
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= minimum_norm:
        return None
    return vector / norm


def build_torso_reference(
    left_shoulder,
    right_shoulder,
    left_hip,
    right_hip,
):
    """Construye una base ortonormal ligada al torso.

    body_right:
        Desde el hombro izquierdo hacia el derecho.

    body_up:
        Desde el centro de las caderas hacia el centro de hombros.

    body_forward:
        Perpendicular al plano torso, calculado con producto cruz.

    La base se recalcula en cada frame para seguir rotaciones e
    inclinaciones moderadas del torso.
    """
    shoulder_center = 0.5 * (left_shoulder + right_shoulder)
    hip_center = 0.5 * (left_hip + right_hip)

    body_right = normalize_vector(right_shoulder - left_shoulder)
    body_up_approx = normalize_vector(shoulder_center - hip_center)

    if body_right is None or body_up_approx is None:
        return None

    # Y corporal: perpendicular al eje lateral, pero contenido
    # en el plano definido por el torso.
    body_forward = normalize_vector(np.cross(body_right, body_up_approx))
    if body_forward is None:
        return None

    # Re-ortogonalizar el eje vertical para reducir ruido y sesgo.
    body_up = normalize_vector(np.cross(body_forward, body_right))
    if body_up is None:
        return None

    # Matriz cuyas columnas son los ejes corporales expresados
    # en coordenadas de camara.
    body_rotation = np.column_stack(
        (body_right, body_up, body_forward)
    ).astype(np.float32)

    return {
        "shoulder_center": shoulder_center.astype(np.float32),
        "hip_center": hip_center.astype(np.float32),
        "body_right": body_right.astype(np.float32),
        "body_up": body_up.astype(np.float32),
        "body_forward": body_forward.astype(np.float32),
        "body_rotation": body_rotation,
    }


def transform_vector_to_body(vector, body_right, body_up, body_forward):
    """Proyecta un vector de camara sobre los ejes del torso."""
    vector = np.asarray(vector, dtype=np.float32)
    return np.asarray(
        [
            np.dot(vector, body_right),
            np.dot(vector, body_up),
            np.dot(vector, body_forward),
        ],
        dtype=np.float32,
    )


def extract_hand_control_points(hand_landmarks):
    """Extrae los puntos usados para calcular apertura de la garra."""
    return {
        index: normalized_landmark_to_array(hand_landmarks[index])
        for index in HAND_CONTROL_INDICES
    }


def smooth_hand_control_points(previous_points, current_points, alpha):
    """Suaviza solamente los puntos usados para la garra."""
    if previous_points is None:
        return {
            index: point.copy()
            for index, point in current_points.items()
        }

    return {
        index: smooth_point(
            previous_points.get(index),
            point,
            alpha,
        )
        for index, point in current_points.items()
    }


def calculate_raw_hand_openness(control_points):
    """Calcula una medida continua de apertura sin calibrar."""
    wrist = control_points[HAND_WRIST]
    thumb_tip = control_points[THUMB_TIP]
    index_mcp = control_points[INDEX_MCP]
    index_tip = control_points[INDEX_TIP]
    middle_mcp = control_points[MIDDLE_MCP]
    middle_tip = control_points[MIDDLE_TIP]

    palm_size = float(np.linalg.norm(middle_mcp - wrist))
    if palm_size <= 1e-6:
        return None

    thumb_index_distance = (
        np.linalg.norm(thumb_tip - index_tip) / palm_size
    )
    index_extension = (
        np.linalg.norm(index_tip - index_mcp) / palm_size
    )
    middle_extension = (
        np.linalg.norm(middle_tip - middle_mcp) / palm_size
    )

    return float(
        0.40 * thumb_index_distance
        + 0.30 * index_extension
        + 0.30 * middle_extension
    )


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
        use_smoothing=True,
    ):
        self.pose_detector = pose_detector
        self.hand_detector = hand_detector
        self.target_arm = target_arm.lower()
        self.use_smoothing = use_smoothing

        self.smoothing_alpha = config.get("SMOOTHING_ALPHA", 0.35)
        self.max_hand_distance = config.get(
            "MAX_HAND_WRIST_DISTANCE",
            0.30,
        )
        self.visibility_threshold = config.get(
            "LANDMARK_VISIBILITY_THRESHOLD",
            0.20,
        )

        self.arm_lost_frame_tolerance = config.get(
            "ARM_LOST_FRAME_TOLERANCE",
            config.get("LOST_FRAME_TOLERANCE", 5),
        )
        self.hand_lost_frame_tolerance = config.get(
            "HAND_LOST_FRAME_TOLERANCE",
            config.get("LOST_FRAME_TOLERANCE", 5),
        )
        self.torso_lost_frame_tolerance = config.get(
            "TORSO_LOST_FRAME_TOLERANCE",
            config.get("LOST_FRAME_TOLERANCE", 5),
        )

        self.draw_arm = config.get("DRAW_ARM", True)
        self.draw_hand = config.get("DRAW_HAND", True)
        self.draw_full_hand = config.get("DRAW_FULL_HAND", False)
        self.draw_torso = config.get("DRAW_TORSO", True)
        self.draw_torso_axes = config.get("DRAW_TORSO_AXES", True)
        self.torso_axis_length = float(
            config.get("TORSO_AXIS_LENGTH", 0.12)
        )

        if self.target_arm == "left":
            self.shoulder_idx = POSE_LEFT_SHOULDER
            self.elbow_idx = POSE_LEFT_ELBOW
            self.wrist_idx = POSE_LEFT_WRIST
        elif self.target_arm == "right":
            self.shoulder_idx = POSE_RIGHT_SHOULDER
            self.elbow_idx = POSE_RIGHT_ELBOW
            self.wrist_idx = POSE_RIGHT_WRIST
        else:
            raise ValueError('TARGET_ARM debe ser "left" o "right".')

        self.hand_wrist_idx = HAND_WRIST

        # Brazo.
        self.prev_shoulder = None
        self.prev_elbow = None
        self.prev_wrist = None
        self.arm_lost_frames = 0

        # Torso.
        self.prev_left_shoulder = None
        self.prev_right_shoulder = None
        self.prev_left_hip = None
        self.prev_right_hip = None
        self.last_torso_reference = None
        self.torso_lost_frames = 0

        # Mano.
        self.prev_hand_control = None
        self.last_full_hand = None
        self.last_hand_label = "Unknown"
        self.last_hand_distance = None
        self.last_raw_hand_openness = None
        self.hand_lost_frames = 0

        # Tiempos.
        self.pose_time_ms = 0.0
        self.hand_time_ms = 0.0
        self.postprocess_time_ms = 0.0

    def reset_smoothing(self):
        """Elimina los estados temporales del procesador."""
        self.prev_shoulder = None
        self.prev_elbow = None
        self.prev_wrist = None
        self.arm_lost_frames = 0

        self.prev_left_shoulder = None
        self.prev_right_shoulder = None
        self.prev_left_hip = None
        self.prev_right_hip = None
        self.last_torso_reference = None
        self.torso_lost_frames = 0

        self.prev_hand_control = None
        self.last_full_hand = None
        self.last_hand_label = "Unknown"
        self.last_hand_distance = None
        self.last_raw_hand_openness = None
        self.hand_lost_frames = 0

    def process_frame(self, mp_image, frame, timestamp_ms):
        height, width, _ = frame.shape

        pose_start = time.perf_counter()
        pose_result = self.pose_detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )
        self.pose_time_ms = (
            time.perf_counter() - pose_start
        ) * 1000.0

        hand_start = time.perf_counter()
        hand_result = self.hand_detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )
        self.hand_time_ms = (
            time.perf_counter() - hand_start
        ) * 1000.0

        postprocess_start = time.perf_counter()

        arm_detected_current = False
        hand_detected_current = False
        torso_detected_current = False

        shoulder_coords = None
        elbow_coords = None
        wrist_coords = None

        left_shoulder_coords = None
        right_shoulder_coords = None
        left_hip_coords = None
        right_hip_coords = None
        torso_reference = None

        full_hand_coords = None
        hand_control_points = None
        hand_label = "Unknown"
        hand_distance = None
        raw_hand_openness = None

        if pose_result.pose_landmarks:
            pose_landmarks = pose_result.pose_landmarks[0]

            # ------------------------------------------------
            # TORSO DEL FRAME ACTUAL
            # ------------------------------------------------
            left_shoulder_landmark = pose_landmarks[POSE_LEFT_SHOULDER]
            right_shoulder_landmark = pose_landmarks[POSE_RIGHT_SHOULDER]
            left_hip_landmark = pose_landmarks[POSE_LEFT_HIP]
            right_hip_landmark = pose_landmarks[POSE_RIGHT_HIP]

            torso_landmarks = (
                left_shoulder_landmark,
                right_shoulder_landmark,
                left_hip_landmark,
                right_hip_landmark,
            )

            if all(
                landmark_is_visible(
                    landmark,
                    self.visibility_threshold,
                )
                for landmark in torso_landmarks
            ):
                left_shoulder_coords = normalized_landmark_to_array(
                    left_shoulder_landmark
                )
                right_shoulder_coords = normalized_landmark_to_array(
                    right_shoulder_landmark
                )
                left_hip_coords = normalized_landmark_to_array(
                    left_hip_landmark
                )
                right_hip_coords = normalized_landmark_to_array(
                    right_hip_landmark
                )

                if self.use_smoothing:
                    left_shoulder_coords = smooth_point(
                        self.prev_left_shoulder,
                        left_shoulder_coords,
                        self.smoothing_alpha,
                    )
                    right_shoulder_coords = smooth_point(
                        self.prev_right_shoulder,
                        right_shoulder_coords,
                        self.smoothing_alpha,
                    )
                    left_hip_coords = smooth_point(
                        self.prev_left_hip,
                        left_hip_coords,
                        self.smoothing_alpha,
                    )
                    right_hip_coords = smooth_point(
                        self.prev_right_hip,
                        right_hip_coords,
                        self.smoothing_alpha,
                    )

                torso_reference = build_torso_reference(
                    left_shoulder_coords,
                    right_shoulder_coords,
                    left_hip_coords,
                    right_hip_coords,
                )

                if torso_reference is not None:
                    torso_detected_current = True
                    self.prev_left_shoulder = left_shoulder_coords.copy()
                    self.prev_right_shoulder = right_shoulder_coords.copy()
                    self.prev_left_hip = left_hip_coords.copy()
                    self.prev_right_hip = right_hip_coords.copy()
                    self.last_torso_reference = {
                        key: value.copy()
                        for key, value in torso_reference.items()
                    }
                    self.torso_lost_frames = 0

            # ------------------------------------------------
            # BRAZO OBJETIVO DEL FRAME ACTUAL
            # ------------------------------------------------
            shoulder_landmark = pose_landmarks[self.shoulder_idx]
            elbow_landmark = pose_landmarks[self.elbow_idx]
            wrist_landmark = pose_landmarks[self.wrist_idx]

            if (
                landmark_is_visible(
                    shoulder_landmark,
                    self.visibility_threshold,
                )
                and landmark_is_visible(
                    elbow_landmark,
                    self.visibility_threshold,
                )
                and landmark_is_visible(
                    wrist_landmark,
                    self.visibility_threshold,
                )
            ):
                arm_detected_current = True
                shoulder_coords = normalized_landmark_to_array(
                    shoulder_landmark
                )
                elbow_coords = normalized_landmark_to_array(
                    elbow_landmark
                )
                wrist_coords = normalized_landmark_to_array(
                    wrist_landmark
                )

                if self.use_smoothing:
                    shoulder_coords = smooth_point(
                        self.prev_shoulder,
                        shoulder_coords,
                        self.smoothing_alpha,
                    )
                    elbow_coords = smooth_point(
                        self.prev_elbow,
                        elbow_coords,
                        self.smoothing_alpha,
                    )
                    wrist_coords = smooth_point(
                        self.prev_wrist,
                        wrist_coords,
                        self.smoothing_alpha,
                    )

                self.prev_shoulder = shoulder_coords.copy()
                self.prev_elbow = elbow_coords.copy()
                self.prev_wrist = wrist_coords.copy()
                self.arm_lost_frames = 0

                (
                    selected_hand,
                    hand_label,
                    hand_distance,
                ) = select_hand_near_pose_wrist(
                    hand_result,
                    wrist_landmark,
                    self.max_hand_distance,
                    self.hand_wrist_idx,
                )

                if selected_hand is not None:
                    hand_detected_current = True
                    full_hand_coords = np.asarray(
                        [
                            normalized_landmark_to_array(landmark)
                            for landmark in selected_hand
                        ],
                        dtype=np.float32,
                    )
                    hand_control_points = extract_hand_control_points(
                        selected_hand
                    )

                    if self.use_smoothing:
                        hand_control_points = smooth_hand_control_points(
                            self.prev_hand_control,
                            hand_control_points,
                            self.smoothing_alpha,
                        )

                    raw_hand_openness = calculate_raw_hand_openness(
                        hand_control_points
                    )

                    self.prev_hand_control = {
                        index: point.copy()
                        for index, point in hand_control_points.items()
                    }
                    self.last_full_hand = full_hand_coords.copy()
                    self.last_hand_label = hand_label
                    self.last_hand_distance = hand_distance
                    self.last_raw_hand_openness = raw_hand_openness
                    self.hand_lost_frames = 0

        # ----------------------------------------------------
        # TOLERANCIAS DE PERDIDA
        # ----------------------------------------------------
        if not torso_detected_current:
            self.torso_lost_frames += 1
            if (
                self.torso_lost_frames
                <= self.torso_lost_frame_tolerance
                and self.last_torso_reference is not None
            ):
                torso_reference = {
                    key: value.copy()
                    for key, value in self.last_torso_reference.items()
                }
                left_shoulder_coords = self.prev_left_shoulder.copy()
                right_shoulder_coords = self.prev_right_shoulder.copy()
                left_hip_coords = self.prev_left_hip.copy()
                right_hip_coords = self.prev_right_hip.copy()
            else:
                torso_reference = None
                left_shoulder_coords = None
                right_shoulder_coords = None
                left_hip_coords = None
                right_hip_coords = None

        if not arm_detected_current:
            self.arm_lost_frames += 1
            if (
                self.arm_lost_frames <= self.arm_lost_frame_tolerance
                and self.prev_shoulder is not None
                and self.prev_elbow is not None
                and self.prev_wrist is not None
            ):
                shoulder_coords = self.prev_shoulder.copy()
                elbow_coords = self.prev_elbow.copy()
                wrist_coords = self.prev_wrist.copy()
            else:
                shoulder_coords = None
                elbow_coords = None
                wrist_coords = None

        if not hand_detected_current:
            self.hand_lost_frames += 1
            if (
                self.hand_lost_frames <= self.hand_lost_frame_tolerance
                and self.prev_hand_control is not None
            ):
                hand_control_points = {
                    index: point.copy()
                    for index, point in self.prev_hand_control.items()
                }
                full_hand_coords = (
                    self.last_full_hand.copy()
                    if self.last_full_hand is not None
                    else None
                )
                hand_label = self.last_hand_label
                hand_distance = self.last_hand_distance
                raw_hand_openness = self.last_raw_hand_openness
            else:
                hand_control_points = None
                full_hand_coords = None
                hand_label = "Unknown"
                hand_distance = None
                raw_hand_openness = None

        arm_tracking_valid = all(
            point is not None
            for point in (
                shoulder_coords,
                elbow_coords,
                wrist_coords,
            )
        )
        hand_tracking_valid = hand_control_points is not None
        torso_reference_valid = torso_reference is not None

        arm_using_previous = (
            arm_tracking_valid and not arm_detected_current
        )
        hand_using_previous = (
            hand_tracking_valid and not hand_detected_current
        )
        torso_using_previous = (
            torso_reference_valid and not torso_detected_current
        )

        # Vectores del brazo expresados en la referencia corporal.
        upper_arm_body = None
        forearm_body = None
        if arm_tracking_valid and torso_reference_valid:
            upper_arm_body = transform_vector_to_body(
                elbow_coords - shoulder_coords,
                torso_reference["body_right"],
                torso_reference["body_up"],
                torso_reference["body_forward"],
            )
            forearm_body = transform_vector_to_body(
                wrist_coords - elbow_coords,
                torso_reference["body_right"],
                torso_reference["body_up"],
                torso_reference["body_forward"],
            )

        # ----------------------------------------------------
        # DIBUJO
        # ----------------------------------------------------
        if self.draw_torso and torso_reference_valid:
            torso_color = (
                (0, 165, 255)
                if torso_using_previous
                else (180, 180, 180)
            )
            draw_line(
                frame,
                left_shoulder_coords,
                right_shoulder_coords,
                width,
                height,
                torso_color,
                thickness=2,
            )
            draw_line(
                frame,
                left_hip_coords,
                right_hip_coords,
                width,
                height,
                torso_color,
                thickness=2,
            )
            draw_line(
                frame,
                torso_reference["shoulder_center"],
                torso_reference["hip_center"],
                width,
                height,
                torso_color,
                thickness=2,
            )

            for point in (
                left_shoulder_coords,
                right_shoulder_coords,
                left_hip_coords,
                right_hip_coords,
            ):
                draw_point(
                    frame,
                    point,
                    width,
                    height,
                    torso_color,
                    radius=5,
                )

            if self.draw_torso_axes:
                origin = torso_reference["shoulder_center"]
                axis_length = self.torso_axis_length
                draw_line(
                    frame,
                    origin,
                    origin + axis_length * torso_reference["body_right"],
                    width,
                    height,
                    (255, 0, 0),
                    thickness=3,
                )
                draw_line(
                    frame,
                    origin,
                    origin + axis_length * torso_reference["body_up"],
                    width,
                    height,
                    (0, 255, 0),
                    thickness=3,
                )
                draw_line(
                    frame,
                    origin,
                    origin + axis_length * torso_reference["body_forward"],
                    width,
                    height,
                    (0, 0, 255),
                    thickness=3,
                )

        if self.draw_arm and arm_tracking_valid:
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
                thickness=4,
            )
            draw_line(
                frame,
                elbow_coords,
                wrist_coords,
                width,
                height,
                arm_color,
                thickness=4,
            )
            draw_point(
                frame,
                shoulder_coords,
                width,
                height,
                (255, 0, 0),
                radius=8,
            )
            draw_point(
                frame,
                elbow_coords,
                width,
                height,
                (0, 165, 255),
                radius=8,
            )
            draw_point(
                frame,
                wrist_coords,
                width,
                height,
                (0, 0, 255),
                radius=8,
            )

        if self.draw_hand and hand_tracking_valid:
            hand_color = (
                (0, 165, 255)
                if hand_using_previous
                else (255, 0, 255)
            )

            if self.draw_full_hand and full_hand_coords is not None:
                for start_index, end_index in HAND_CONNECTIONS:
                    draw_line(
                        frame,
                        full_hand_coords[start_index],
                        full_hand_coords[end_index],
                        width,
                        height,
                        hand_color,
                        thickness=2,
                    )
                for index, point in enumerate(full_hand_coords):
                    draw_point(
                        frame,
                        point,
                        width,
                        height,
                        (0, 0, 255)
                        if index == HAND_WRIST
                        else (0, 255, 255),
                        radius=4,
                    )
            else:
                for start_index, end_index in HAND_CONTROL_CONNECTIONS:
                    draw_line(
                        frame,
                        hand_control_points[start_index],
                        hand_control_points[end_index],
                        width,
                        height,
                        hand_color,
                        thickness=2,
                    )
                for index, point in hand_control_points.items():
                    draw_point(
                        frame,
                        point,
                        width,
                        height,
                        (0, 0, 255)
                        if index == HAND_WRIST
                        else (0, 255, 255),
                        radius=5,
                    )

        self.postprocess_time_ms = (
            time.perf_counter() - postprocess_start
        ) * 1000.0

        result = {
            "arm_detected": arm_tracking_valid,
            "hand_detected": hand_tracking_valid,
            "torso_reference_valid": torso_reference_valid,
            "arm_detected_current": arm_detected_current,
            "hand_detected_current": hand_detected_current,
            "torso_detected_current": torso_detected_current,
            "arm_using_previous": arm_using_previous,
            "hand_using_previous": hand_using_previous,
            "torso_using_previous": torso_using_previous,
            "shoulder": shoulder_coords,
            "elbow": elbow_coords,
            "wrist": wrist_coords,
            "left_shoulder": left_shoulder_coords,
            "right_shoulder": right_shoulder_coords,
            "left_hip": left_hip_coords,
            "right_hip": right_hip_coords,
            "hand": full_hand_coords,
            "hand_control": hand_control_points,
            "raw_hand_openness": raw_hand_openness,
            "hand_label": hand_label,
            "distance": hand_distance,
            "upper_arm_body": upper_arm_body,
            "forearm_body": forearm_body,
            "arm_lost_frames": self.arm_lost_frames,
            "hand_lost_frames": self.hand_lost_frames,
            "torso_lost_frames": self.torso_lost_frames,
            "pose_time_ms": self.pose_time_ms,
            "hand_time_ms": self.hand_time_ms,
            "postprocess_time_ms": self.postprocess_time_ms,
        }

        if torso_reference_valid:
            result.update(torso_reference)
        else:
            result.update({
                "shoulder_center": None,
                "hip_center": None,
                "body_right": None,
                "body_up": None,
                "body_forward": None,
                "body_rotation": None,
            })

        return result
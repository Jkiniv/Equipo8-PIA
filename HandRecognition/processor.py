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


# Pose Landmarker indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

# Hand Landmarker indices used by the gripper
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


def world_landmark_to_array(landmark):
    """Convert a MediaPipe world landmark to a float32 xyz array."""
    return np.asarray(
        [landmark.x, landmark.y, landmark.z],
        dtype=np.float32,
    )


def normalize_vector(vector, minimum_norm=1e-6):
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
    """Build an orthonormal torso reference from four 3D points."""
    shoulder_center = 0.5 * (left_shoulder + right_shoulder)
    hip_center = 0.5 * (left_hip + right_hip)

    body_right = normalize_vector(right_shoulder - left_shoulder)
    body_up_approx = normalize_vector(shoulder_center - hip_center)
    if body_right is None or body_up_approx is None:
        return None

    body_forward = normalize_vector(np.cross(body_right, body_up_approx))
    if body_forward is None:
        return None

    body_up = normalize_vector(np.cross(body_forward, body_right))
    if body_up is None:
        return None

    return {
        "shoulder_center": shoulder_center.astype(np.float32),
        "hip_center": hip_center.astype(np.float32),
        "body_right": body_right.astype(np.float32),
        "body_up": body_up.astype(np.float32),
        "body_forward": body_forward.astype(np.float32),
        "body_rotation": np.column_stack(
            (body_right, body_up, body_forward)
        ).astype(np.float32),
    }


def transform_vector_to_body(vector, torso_reference):
    vector = np.asarray(vector, dtype=np.float32)
    return np.asarray(
        [
            np.dot(vector, torso_reference["body_right"]),
            np.dot(vector, torso_reference["body_up"]),
            np.dot(vector, torso_reference["body_forward"]),
        ],
        dtype=np.float32,
    )


def extract_hand_control_points(hand_landmarks):
    return {
        index: normalized_landmark_to_array(hand_landmarks[index])
        for index in HAND_CONTROL_INDICES
    }


def smooth_hand_control_points(previous_points, current_points, alpha):
    if previous_points is None:
        return {
            index: point.copy()
            for index, point in current_points.items()
        }

    return {
        index: smooth_point(previous_points.get(index), point, alpha)
        for index, point in current_points.items()
    }


def calculate_raw_hand_openness(control_points):
    wrist = control_points[HAND_WRIST]
    thumb_tip = control_points[THUMB_TIP]
    index_mcp = control_points[INDEX_MCP]
    index_tip = control_points[INDEX_TIP]
    middle_mcp = control_points[MIDDLE_MCP]
    middle_tip = control_points[MIDDLE_TIP]

    palm_size = float(np.linalg.norm(middle_mcp - wrist))
    if palm_size <= 1e-6:
        return None

    thumb_index = np.linalg.norm(thumb_tip - index_tip) / palm_size
    index_extension = np.linalg.norm(index_tip - index_mcp) / palm_size
    middle_extension = np.linalg.norm(middle_tip - middle_mcp) / palm_size

    return float(
        0.40 * thumb_index
        + 0.30 * index_extension
        + 0.30 * middle_extension
    )


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
        self.max_hand_distance = config.get("MAX_HAND_WRIST_DISTANCE", 0.30)
        self.visibility_threshold = config.get(
            "LANDMARK_VISIBILITY_THRESHOLD",
            0.20,
        )

        default_tolerance = config.get("LOST_FRAME_TOLERANCE", 5)
        self.arm_tolerance = config.get(
            "ARM_LOST_FRAME_TOLERANCE",
            default_tolerance,
        )
        self.hand_tolerance = config.get(
            "HAND_LOST_FRAME_TOLERANCE",
            default_tolerance,
        )
        self.torso_tolerance = config.get(
            "TORSO_LOST_FRAME_TOLERANCE",
            default_tolerance,
        )

        self.draw_arm = config.get("DRAW_ARM", True)
        self.draw_hand = config.get("DRAW_HAND", True)
        self.draw_full_hand = config.get("DRAW_FULL_HAND", False)
        self.draw_torso = config.get("DRAW_TORSO", True)
        self.draw_torso_axes = config.get("DRAW_TORSO_AXES", True)
        self.torso_axis_length = float(config.get("TORSO_AXIS_LENGTH", 0.12))

        if self.target_arm == "left":
            self.shoulder_idx = LEFT_SHOULDER
            self.elbow_idx = LEFT_ELBOW
            self.wrist_idx = LEFT_WRIST
        elif self.target_arm == "right":
            self.shoulder_idx = RIGHT_SHOULDER
            self.elbow_idx = RIGHT_ELBOW
            self.wrist_idx = RIGHT_WRIST
        else:
            raise ValueError('TARGET_ARM debe ser "left" o "right".')

        self.hand_wrist_idx = HAND_WRIST
        self.pose_time_ms = 0.0
        self.hand_time_ms = 0.0
        self.postprocess_time_ms = 0.0
        self.reset_smoothing()

    def reset_smoothing(self):
        self.prev_shoulder = None
        self.prev_elbow = None
        self.prev_wrist = None
        self.prev_shoulder_world = None
        self.prev_elbow_world = None
        self.prev_wrist_world = None
        self.arm_lost_frames = 0

        self.prev_left_shoulder = None
        self.prev_right_shoulder = None
        self.prev_left_hip = None
        self.prev_right_hip = None
        self.prev_left_shoulder_world = None
        self.prev_right_shoulder_world = None
        self.prev_left_hip_world = None
        self.prev_right_hip_world = None
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
        pose_result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
        self.pose_time_ms = (time.perf_counter() - pose_start) * 1000.0

        hand_start = time.perf_counter()
        hand_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
        self.hand_time_ms = (time.perf_counter() - hand_start) * 1000.0

        post_start = time.perf_counter()

        arm_current = False
        hand_current = False
        torso_current = False

        shoulder = elbow = wrist = None
        shoulder_world = elbow_world = wrist_world = None
        left_shoulder = right_shoulder = left_hip = right_hip = None
        left_shoulder_world = right_shoulder_world = None
        left_hip_world = right_hip_world = None
        torso_reference = None

        full_hand = None
        hand_control = None
        hand_label = "Unknown"
        hand_distance = None
        raw_hand_openness = None

        if pose_result.pose_landmarks and pose_result.pose_world_landmarks:
            pose = pose_result.pose_landmarks[0]
            world = pose_result.pose_world_landmarks[0]

            torso_landmarks = (
                pose[LEFT_SHOULDER],
                pose[RIGHT_SHOULDER],
                pose[LEFT_HIP],
                pose[RIGHT_HIP],
            )

            if all(
                landmark_is_visible(item, self.visibility_threshold)
                for item in torso_landmarks
            ):
                left_shoulder = normalized_landmark_to_array(pose[LEFT_SHOULDER])
                right_shoulder = normalized_landmark_to_array(pose[RIGHT_SHOULDER])
                left_hip = normalized_landmark_to_array(pose[LEFT_HIP])
                right_hip = normalized_landmark_to_array(pose[RIGHT_HIP])

                left_shoulder_world = world_landmark_to_array(world[LEFT_SHOULDER])
                right_shoulder_world = world_landmark_to_array(world[RIGHT_SHOULDER])
                left_hip_world = world_landmark_to_array(world[LEFT_HIP])
                right_hip_world = world_landmark_to_array(world[RIGHT_HIP])

                if self.use_smoothing:
                    left_shoulder = smooth_point(
                        self.prev_left_shoulder, left_shoulder, self.smoothing_alpha
                    )
                    right_shoulder = smooth_point(
                        self.prev_right_shoulder, right_shoulder, self.smoothing_alpha
                    )
                    left_hip = smooth_point(
                        self.prev_left_hip, left_hip, self.smoothing_alpha
                    )
                    right_hip = smooth_point(
                        self.prev_right_hip, right_hip, self.smoothing_alpha
                    )
                    left_shoulder_world = smooth_point(
                        self.prev_left_shoulder_world,
                        left_shoulder_world,
                        self.smoothing_alpha,
                    )
                    right_shoulder_world = smooth_point(
                        self.prev_right_shoulder_world,
                        right_shoulder_world,
                        self.smoothing_alpha,
                    )
                    left_hip_world = smooth_point(
                        self.prev_left_hip_world,
                        left_hip_world,
                        self.smoothing_alpha,
                    )
                    right_hip_world = smooth_point(
                        self.prev_right_hip_world,
                        right_hip_world,
                        self.smoothing_alpha,
                    )

                torso_reference = build_torso_reference(
                    left_shoulder_world,
                    right_shoulder_world,
                    left_hip_world,
                    right_hip_world,
                )

                if torso_reference is not None:
                    torso_current = True
                    self.prev_left_shoulder = left_shoulder.copy()
                    self.prev_right_shoulder = right_shoulder.copy()
                    self.prev_left_hip = left_hip.copy()
                    self.prev_right_hip = right_hip.copy()
                    self.prev_left_shoulder_world = left_shoulder_world.copy()
                    self.prev_right_shoulder_world = right_shoulder_world.copy()
                    self.prev_left_hip_world = left_hip_world.copy()
                    self.prev_right_hip_world = right_hip_world.copy()
                    self.last_torso_reference = {
                        key: value.copy()
                        for key, value in torso_reference.items()
                    }
                    self.torso_lost_frames = 0

            shoulder_landmark = pose[self.shoulder_idx]
            elbow_landmark = pose[self.elbow_idx]
            wrist_landmark = pose[self.wrist_idx]

            if (
                landmark_is_visible(shoulder_landmark, self.visibility_threshold)
                and landmark_is_visible(elbow_landmark, self.visibility_threshold)
                and landmark_is_visible(wrist_landmark, self.visibility_threshold)
            ):
                arm_current = True
                shoulder = normalized_landmark_to_array(shoulder_landmark)
                elbow = normalized_landmark_to_array(elbow_landmark)
                wrist = normalized_landmark_to_array(wrist_landmark)

                shoulder_world = world_landmark_to_array(world[self.shoulder_idx])
                elbow_world = world_landmark_to_array(world[self.elbow_idx])
                wrist_world = world_landmark_to_array(world[self.wrist_idx])

                if self.use_smoothing:
                    shoulder = smooth_point(
                        self.prev_shoulder, shoulder, self.smoothing_alpha
                    )
                    elbow = smooth_point(self.prev_elbow, elbow, self.smoothing_alpha)
                    wrist = smooth_point(self.prev_wrist, wrist, self.smoothing_alpha)
                    shoulder_world = smooth_point(
                        self.prev_shoulder_world,
                        shoulder_world,
                        self.smoothing_alpha,
                    )
                    elbow_world = smooth_point(
                        self.prev_elbow_world,
                        elbow_world,
                        self.smoothing_alpha,
                    )
                    wrist_world = smooth_point(
                        self.prev_wrist_world,
                        wrist_world,
                        self.smoothing_alpha,
                    )

                self.prev_shoulder = shoulder.copy()
                self.prev_elbow = elbow.copy()
                self.prev_wrist = wrist.copy()
                self.prev_shoulder_world = shoulder_world.copy()
                self.prev_elbow_world = elbow_world.copy()
                self.prev_wrist_world = wrist_world.copy()
                self.arm_lost_frames = 0

                selected_hand, hand_label, hand_distance = (
                    select_hand_near_pose_wrist(
                        hand_result,
                        wrist_landmark,
                        self.max_hand_distance,
                        self.hand_wrist_idx,
                    )
                )

                if selected_hand is not None:
                    hand_current = True
                    full_hand = np.asarray(
                        [
                            normalized_landmark_to_array(item)
                            for item in selected_hand
                        ],
                        dtype=np.float32,
                    )
                    hand_control = extract_hand_control_points(selected_hand)
                    if self.use_smoothing:
                        hand_control = smooth_hand_control_points(
                            self.prev_hand_control,
                            hand_control,
                            self.smoothing_alpha,
                        )
                    raw_hand_openness = calculate_raw_hand_openness(hand_control)
                    self.prev_hand_control = {
                        index: point.copy()
                        for index, point in hand_control.items()
                    }
                    self.last_full_hand = full_hand.copy()
                    self.last_hand_label = hand_label
                    self.last_hand_distance = hand_distance
                    self.last_raw_hand_openness = raw_hand_openness
                    self.hand_lost_frames = 0

        if not torso_current:
            self.torso_lost_frames += 1
            if (
                self.torso_lost_frames <= self.torso_tolerance
                and self.last_torso_reference is not None
            ):
                torso_reference = {
                    key: value.copy()
                    for key, value in self.last_torso_reference.items()
                }
                left_shoulder = self.prev_left_shoulder.copy()
                right_shoulder = self.prev_right_shoulder.copy()
                left_hip = self.prev_left_hip.copy()
                right_hip = self.prev_right_hip.copy()
                left_shoulder_world = self.prev_left_shoulder_world.copy()
                right_shoulder_world = self.prev_right_shoulder_world.copy()
                left_hip_world = self.prev_left_hip_world.copy()
                right_hip_world = self.prev_right_hip_world.copy()
            else:
                torso_reference = None

        if not arm_current:
            self.arm_lost_frames += 1
            if (
                self.arm_lost_frames <= self.arm_tolerance
                and self.prev_shoulder is not None
            ):
                shoulder = self.prev_shoulder.copy()
                elbow = self.prev_elbow.copy()
                wrist = self.prev_wrist.copy()
                shoulder_world = self.prev_shoulder_world.copy()
                elbow_world = self.prev_elbow_world.copy()
                wrist_world = self.prev_wrist_world.copy()
            else:
                shoulder = elbow = wrist = None
                shoulder_world = elbow_world = wrist_world = None

        if not hand_current:
            self.hand_lost_frames += 1
            if (
                self.hand_lost_frames <= self.hand_tolerance
                and self.prev_hand_control is not None
            ):
                hand_control = {
                    index: point.copy()
                    for index, point in self.prev_hand_control.items()
                }
                full_hand = (
                    self.last_full_hand.copy()
                    if self.last_full_hand is not None
                    else None
                )
                hand_label = self.last_hand_label
                hand_distance = self.last_hand_distance
                raw_hand_openness = self.last_raw_hand_openness
            else:
                hand_control = None
                full_hand = None
                hand_label = "Unknown"
                hand_distance = None
                raw_hand_openness = None

        arm_valid = all(item is not None for item in (shoulder, elbow, wrist))
        hand_valid = hand_control is not None
        torso_valid = torso_reference is not None

        arm_previous = arm_valid and not arm_current
        hand_previous = hand_valid and not hand_current
        torso_previous = torso_valid and not torso_current

        upper_arm_body = None
        forearm_body = None
        if (
            arm_valid
            and torso_valid
            and shoulder_world is not None
            and elbow_world is not None
            and wrist_world is not None
        ):
            upper_arm_body = transform_vector_to_body(
                elbow_world - shoulder_world,
                torso_reference,
            )
            forearm_body = transform_vector_to_body(
                wrist_world - elbow_world,
                torso_reference,
            )

        if self.draw_torso and torso_valid:
            torso_color = (0, 165, 255) if torso_previous else (180, 180, 180)
            draw_line(
                frame, left_shoulder, right_shoulder,
                width, height, torso_color, thickness=2,
            )
            draw_line(
                frame, left_hip, right_hip,
                width, height, torso_color, thickness=2,
            )
            image_reference = build_torso_reference(
                left_shoulder,
                right_shoulder,
                left_hip,
                right_hip,
            )
            if image_reference is not None:
                draw_line(
                    frame,
                    image_reference["shoulder_center"],
                    image_reference["hip_center"],
                    width,
                    height,
                    torso_color,
                    thickness=2,
                )
                if self.draw_torso_axes:
                    origin = image_reference["shoulder_center"]
                    length = self.torso_axis_length
                    for vector_name, color in (
                        ("body_right", (255, 0, 0)),
                        ("body_up", (0, 255, 0)),
                        ("body_forward", (0, 0, 255)),
                    ):
                        draw_line(
                            frame,
                            origin,
                            origin + length * image_reference[vector_name],
                            width,
                            height,
                            color,
                            thickness=3,
                        )

        if self.draw_arm and arm_valid:
            arm_color = (0, 165, 255) if arm_previous else (0, 255, 0)
            draw_line(frame, shoulder, elbow, width, height, arm_color, thickness=4)
            draw_line(frame, elbow, wrist, width, height, arm_color, thickness=4)
            draw_point(frame, shoulder, width, height, (255, 0, 0), radius=8)
            draw_point(frame, elbow, width, height, (0, 165, 255), radius=8)
            draw_point(frame, wrist, width, height, (0, 0, 255), radius=8)

        if self.draw_hand and hand_valid:
            hand_color = (0, 165, 255) if hand_previous else (255, 0, 255)
            if self.draw_full_hand and full_hand is not None:
                for start_index, end_index in HAND_CONNECTIONS:
                    draw_line(
                        frame,
                        full_hand[start_index],
                        full_hand[end_index],
                        width,
                        height,
                        hand_color,
                        thickness=2,
                    )
                for index, point in enumerate(full_hand):
                    draw_point(
                        frame,
                        point,
                        width,
                        height,
                        (0, 0, 255) if index == HAND_WRIST else (0, 255, 255),
                        radius=4,
                    )
            else:
                for start_index, end_index in HAND_CONTROL_CONNECTIONS:
                    draw_line(
                        frame,
                        hand_control[start_index],
                        hand_control[end_index],
                        width,
                        height,
                        hand_color,
                        thickness=2,
                    )
                for index, point in hand_control.items():
                    draw_point(
                        frame,
                        point,
                        width,
                        height,
                        (0, 0, 255) if index == HAND_WRIST else (0, 255, 255),
                        radius=5,
                    )

        self.postprocess_time_ms = (time.perf_counter() - post_start) * 1000.0

        result = {
            "arm_detected": arm_valid,
            "hand_detected": hand_valid,
            "torso_reference_valid": torso_valid,
            "arm_detected_current": arm_current,
            "hand_detected_current": hand_current,
            "torso_detected_current": torso_current,
            "arm_using_previous": arm_previous,
            "hand_using_previous": hand_previous,
            "torso_using_previous": torso_previous,
            "shoulder": shoulder,
            "elbow": elbow,
            "wrist": wrist,
            "shoulder_world": shoulder_world,
            "elbow_world": elbow_world,
            "wrist_world": wrist_world,
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "left_hip": left_hip,
            "right_hip": right_hip,
            "left_shoulder_world": left_shoulder_world,
            "right_shoulder_world": right_shoulder_world,
            "left_hip_world": left_hip_world,
            "right_hip_world": right_hip_world,
            "upper_arm_body": upper_arm_body,
            "forearm_body": forearm_body,
            "hand": full_hand,
            "hand_control": hand_control,
            "raw_hand_openness": raw_hand_openness,
            "hand_label": hand_label,
            "distance": hand_distance,
            "arm_lost_frames": self.arm_lost_frames,
            "hand_lost_frames": self.hand_lost_frames,
            "torso_lost_frames": self.torso_lost_frames,
            "pose_time_ms": self.pose_time_ms,
            "hand_time_ms": self.hand_time_ms,
            "postprocess_time_ms": self.postprocess_time_ms,
        }

        if torso_valid:
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
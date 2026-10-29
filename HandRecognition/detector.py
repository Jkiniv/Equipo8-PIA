import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def initialize_detectors(
    pose_model_path,
    hand_model_path,
    options_config
):
    """
    Verifica la existencia de los modelos y crea los detectores
    Pose Landmarker y Hand Landmarker de MediaPipe.

    Parameters
    ----------
    pose_model_path:
        Ruta al modelo Pose Landmarker.

    hand_model_path:
        Ruta al modelo Hand Landmarker.

    options_config:
        Diccionario con los parámetros de configuración
        de Pose Landmarker y Hand Landmarker.

    Returns
    -------
    tuple:
        pose_detector, hand_detector
    """

    # ========================================================
    # VERIFICAR ARCHIVOS DE LOS MODELOS
    # ========================================================

    if not os.path.isfile(pose_model_path):
        raise FileNotFoundError(
            "No se encontró el modelo de Pose Landmarker:\n"
            f"{pose_model_path}"
        )

    if not os.path.isfile(hand_model_path):
        raise FileNotFoundError(
            "No se encontró el modelo de Hand Landmarker:\n"
            f"{hand_model_path}"
        )

    # ========================================================
    # LEER Y VALIDAR CONFIGURACIÓN
    # ========================================================

    # El proyecto controla un solo brazo y selecciona una sola
    # mano. Por ello, el valor predeterminado es una mano.
    num_hands = int(
        options_config.get(
            "NUM_HANDS",
            2
        )
    )

    if num_hands < 1:
        raise ValueError(
            "NUM_HANDS debe ser un número entero mayor o igual a 1."
        )

    # ========================================================
    # CONFIGURAR POSE LANDMARKER
    # ========================================================

    pose_base_options = python.BaseOptions(
        model_asset_path=str(pose_model_path)
    )

    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=options_config[
            "POSE_DETECTION_CONFIDENCE"
        ],
        min_pose_presence_confidence=options_config[
            "POSE_PRESENCE_CONFIDENCE"
        ],
        min_tracking_confidence=options_config[
            "POSE_TRACKING_CONFIDENCE"
        ],
        output_segmentation_masks=False,
    )

    # ========================================================
    # CONFIGURAR HAND LANDMARKER
    # ========================================================

    hand_base_options = python.BaseOptions(
        model_asset_path=str(hand_model_path)
    )

    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=options_config[
            "HAND_DETECTION_CONFIDENCE"
        ],
        min_hand_presence_confidence=options_config[
            "HAND_PRESENCE_CONFIDENCE"
        ],
        min_tracking_confidence=options_config[
            "HAND_TRACKING_CONFIDENCE"
        ],
    )

    # ========================================================
    # CREAR DETECTORES
    # ========================================================

    pose_detector = (
        vision.PoseLandmarker.create_from_options(
            pose_options
        )
    )

    hand_detector = (
        vision.HandLandmarker.create_from_options(
            hand_options
        )
    )

    return pose_detector, hand_detector
from typing import Any, Dict

DEBUG: bool = True
MIN_DETECTION_CONFIDENCE: float = 0.9
MIN_TRACKING_CONFIDENCE: float = 0.5
ESC_KEY: int = 27 # ESC key
SHOW_FPS: bool = True
MODEL_COMPLEXITY: int = 1
MAX_NUM_HANDS: int = 1


FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
FRAME_RATE: int = 30

CALIBRATION: Dict[str, Any] = {
	# TODO: Adjust according to the calibration
	'TOLERANCE_FACTOR': 0.5 # Hypothetical
}

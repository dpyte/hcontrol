import cv2


_capture: None | cv2.VideoCapture = None


def initialize_capture_device(obj: str | int | None):
	global _capture
	_capture = cv2.VideoCapture(obj)
	return _capture


def video_properties():
	global _capture
	if _capture is None:
		return None
	if not _capture.isOpened():
		return None
	return {
		'width': int(_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
		'height': int(_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
		'fps': int(_capture.get(cv2.CAP_PROP_FPS)),
		'total_frames': int(_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
	}

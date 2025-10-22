import dataclasses
from enum import IntEnum


class Point(IntEnum):
	"""Hand landmark point enumeration (matches MediaPipe)."""
	Wrist = 0
	THUMB_CMC = 1
	THUMB_MCP = 2
	THUMB_IP = 3
	THUMB_TIP = 4
	INDEX_FINGER_MCP = 5
	INDEX_FINGER_PIP = 6
	INDEX_FINGER_DIP = 7
	INDEX_FINGER_TIP = 8
	MIDDLE_FINGER_MCP = 9
	MIDDLE_FINGER_PIP = 10
	MIDDLE_FINGER_DIP = 11
	MIDDLE_FINGER_TIP = 12
	RING_FINGER_MCP = 13
	RING_FINGER_PIP = 14
	RING_FINGER_DIP = 15
	RING_FINGER_TIP = 16
	PINKY_MCP = 17
	PINKY_PIP = 18
	PINKY_DIP = 19
	PINKY_TIP = 20


@dataclasses.dataclass
class ExtractionPoints:
	"""Extraction points for hand landmarks."""
	palm: float = -1.0
	thumb: float = -1.0
	index: float = -1.0
	middle: float = -1.0
	ring: float = -1.0
	pinky: float = -1.0


IGNORE_Z = True  # Set too False to include Z-axis in calculations
CHANGE_THRESHOLD = 0.03 if IGNORE_Z else 0.07

TRACKED_POINTS = {
	Point.Wrist,
	Point.THUMB_CMC,
	Point.INDEX_FINGER_MCP,
	Point.INDEX_FINGER_PIP,
	Point.MIDDLE_FINGER_MCP,
	Point.MIDDLE_FINGER_PIP,
}

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Deque

from controls.types import Point


class GestureState(Enum):
	"""State machine states for gesture recognition."""
	IDLE = auto()  # No hand detected or outside the activation zone
	READY = auto()  # Hand in activation zone, ready for gestures
	HOVERING = auto()  # Pointer mode - controlling cursor position
	PINCH_START = auto()  # Initial pinch detected
	PINCH_HOLD = auto()  # Pinch maintained
	DRAGGING = auto()  # Pinch + movement = drag
	CLICKING = auto()  # Quick pinch = click
	DOUBLE_CLICK_WAIT = auto()  # Waiting for the second click
	SCROLLING = auto()  # Two-finger scroll mode
	ZOOM = auto()  # Pinch-to-zoom gesture


class GestureType(Enum):
	"""Recognized gesture types."""
	NONE = auto()
	HOVER = auto()  # Move cursor
	CLICK = auto()  # Single click
	DOUBLE_CLICK = auto()  # Double click
	RIGHT_CLICK = auto()  # Hold + release
	DRAG_START = auto()  # Begin dragging
	DRAG_MOVE = auto()  # Continue dragging
	DRAG_END = auto()  # End dragging
	SCROLL = auto()  # Scroll gesture
	ZOOM_IN = auto()  # Zoom in
	ZOOM_OUT = auto()  # Zoom out
	PALM_REST = auto()  # Hand rest position (deactivate)


@dataclass
class GestureEvent:
	"""Represents a detected gesture event."""
	gesture_type: GestureType
	timestamp: float
	position: Optional[Tuple[float, float]] = None
	velocity: Optional[Tuple[float, float]] = None
	confidence: float = 1.0
	metadata: Dict = field(default_factory=dict)


@dataclass
class HandMetrics:
	"""Metrics about hand position and movement."""
	position: Tuple[float, float, float]
	velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
	acceleration: Tuple[float, float, float] = (0.0, 0.0, 0.0)
	pinch_distance: float = 0.0
	palm_orientation: float = 0.0
	hand_openness: float = 0.0
	in_activation_zone: bool = False
	stability_score: float = 0.0


class ActivationZone:
	"""Defines ergonomic activation zones to prevent gorilla arm."""
	def __init__(
		self,
		min_y: float = 0.3,  # Minimum height (30% from bottom)
		max_y: float = 0.7,  # Maximum height (70% from bottom)
		min_x: float = 0.25,  # Left boundary
		max_x: float = 0.75,  # Right boundary
		min_z: float = -0.3,  # Closer to camera
		max_z: float = 0.1  # Further from camera
	):
		self.min_y = min_y
		self.max_y = max_y
		self.min_x = min_x
		self.max_x = max_x
		self.min_z = min_z
		self.max_z = max_z

	def is_in_zone(self, x: float, y: float, z: float) -> bool:
		"""Check if position is within activation zone."""
		return (
			self.min_x <= x <= self.max_x and
			self.min_y <= y <= self.max_y and
			self.min_z <= z <= self.max_z
		)

	def get_zone_status(self, x: float, y: float, z: float) -> str:
		"""Get descriptive status of position relative to zone."""
		if self.is_in_zone(x, y, z):
			return "ACTIVE"
		if y < self.min_y:
			return "TOO_LOW"
		if y > self.max_y:
			return "TOO_HIGH"
		if x < self.min_x:
			return "TOO_LEFT"
		if x > self.max_x:
			return "TOO_RIGHT"
		if z < self.min_z:
			return "TOO_CLOSE"
		return "TOO_FAR"


class GestureRecognizer:
	"""
	Advanced gesture recognizer with state machine and temporal analysis.

	Features:
	- Ergonomic activation zones
	- Temporal smoothing and prediction
	- State-based gesture recognition
	- Configurable sensitivity
	"""

	def __init__(
		self,
		history_size: int = 10,
		pinch_threshold: float = 0.05,
		click_time_threshold: float = 0.3,
		double_click_window: float = 0.5,
		movement_threshold: float = 0.02,
		stability_threshold: float = 0.01
	):
		# Configuration
		self.pinch_threshold = pinch_threshold
		self.click_time_threshold = click_time_threshold
		self.double_click_window = double_click_window
		self.movement_threshold = movement_threshold
		self.stability_threshold = stability_threshold
		# State tracking
		self.current_state = GestureState.IDLE
		self.previous_state = GestureState.IDLE
		self.state_entry_time = time.time()
		# History buffers
		self.position_history: Deque = deque(maxlen=history_size)
		self.pinch_history: Deque = deque(maxlen=history_size)
		self.velocity_history: Deque = deque(maxlen=history_size)
		# Gesture tracking
		self.last_click_time: Optional[float] = None
		self.pinch_start_time: Optional[float] = None
		self.drag_start_position: Optional[Tuple[float, float]] = None
		self.gesture_events: List[GestureEvent] = []

		# Ergonomic features
		self.activation_zone = ActivationZone(
			min_x=0.0, max_x=1.0,
			min_y=0.0, max_y=1.0
		)
		self.rest_timeout = 10.0  # Seconds of inactivity before rest mode
		self.last_activity_time = time.time()

		# Smoothing
		self.smoothed_position: Optional[Tuple[float, float]] = None
		self.smoothing_factor = 0.9  # Higher = more smoothing

	def update(self, landmarks: Dict, frame_width: int, frame_height: int) -> List[GestureEvent]:
		"""
		Process new hand landmarks and return detected gestures.

		Args:
				landmarks: Dictionary of hand landmarks from MediaPipe
				frame_width: Width of the video frame
				frame_height: Height of the video frame

		Returns:
				List of detected gesture events
		"""
		self.gesture_events.clear()

		if not landmarks:
			self._transition_to(GestureState.IDLE)
			return self.gesture_events

		# Calculate hand metrics
		metrics = self._calculate_hand_metrics(landmarks)

		# Update history
		self.position_history.append(metrics.position[:2])
		self.pinch_history.append(metrics.pinch_distance)
		self.velocity_history.append(metrics.velocity)

		# Check activation zone
		if not metrics.in_activation_zone:
			if self.current_state != GestureState.IDLE:
				self._add_event(GestureType.PALM_REST)
			self._transition_to(GestureState.IDLE)
			return self.gesture_events
		self.last_activity_time = time.time()
		self._process_state_machine(metrics)
		return self.gesture_events

	def _calculate_hand_metrics(
		self,
		landmarks: Dict
	) -> HandMetrics:
		"""Calculate comprehensive hand metrics from landmarks."""
		index_tip = landmarks.get(Point.INDEX_FINGER_TIP)
		thumb_tip = landmarks.get(Point.THUMB_TIP)
		wrist = landmarks.get(Point.Wrist)
		middle_tip = landmarks.get(Point.MIDDLE_FINGER_TIP)
		if not all([index_tip, thumb_tip, wrist]):
			return HandMetrics(position=(0, 0, 0), in_activation_zone=False)
		pinch_distance = self._calculate_distance_3d(index_tip, thumb_tip)
		pointer_pos = (index_tip.x, index_tip.y, index_tip.z)
		velocity = (0.0, 0.0, 0.0)
		if len(self.position_history) > 0:
			prev_pos = self.position_history[-1]
			dt = 1 / 30  # Assume 30 FPS
			velocity = (
				(pointer_pos[0] - prev_pos[0]) / dt,
				(pointer_pos[1] - prev_pos[1]) / dt,
				0.0
			)
		hand_openness = 0.0
		if middle_tip:
			hand_openness = self._calculate_distance_3d(index_tip, middle_tip)
		in_zone = self.activation_zone.is_in_zone(
			pointer_pos[0], pointer_pos[1], pointer_pos[2]
		)
		stability = 1.0 - min(1.0, math.sqrt(velocity[0] ** 2 + velocity[1] ** 2) * 10)
		return HandMetrics(
			position=pointer_pos,
			velocity=velocity,
			pinch_distance=pinch_distance,
			hand_openness=hand_openness,
			in_activation_zone=in_zone,
			stability_score=stability
		)

	def _process_state_machine(self, metrics: HandMetrics) -> None:
		"""Process the state machine based on current metrics."""
		is_pinching = metrics.pinch_distance < self.pinch_threshold
		is_moving = math.sqrt(metrics.velocity[0] ** 2 + metrics.velocity[1] ** 2) > self.movement_threshold
		is_stable = metrics.stability_score > 0.8
		current_time = time.time()
		time_in_state = current_time - self.state_entry_time
		if self.current_state == GestureState.IDLE:
			if metrics.in_activation_zone:
				self._transition_to(GestureState.READY)
		elif self.current_state == GestureState.READY:
			if is_pinching:
				self.pinch_start_time = current_time
				self._transition_to(GestureState.PINCH_START)
			elif is_stable:
				self._transition_to(GestureState.HOVERING)
				self._add_hover_event(metrics)
		elif self.current_state == GestureState.HOVERING:
			if is_pinching:
				self.pinch_start_time = current_time
				self._transition_to(GestureState.PINCH_START)
			else:
				self._add_hover_event(metrics)
		elif self.current_state == GestureState.PINCH_START:
			if not is_pinching:
				# Quick release = click
				if time_in_state < self.click_time_threshold:
					self._handle_click(metrics)
				self._transition_to(GestureState.READY)
			elif is_moving:
				# Movement while pinching = drag
				self.drag_start_position = metrics.position[:2]
				self._add_event(GestureType.DRAG_START, metrics.position[:2])
				self._transition_to(GestureState.DRAGGING)
			elif time_in_state > self.click_time_threshold:
				# Long hold = transition to hold state
				self._transition_to(GestureState.PINCH_HOLD)
		elif self.current_state == GestureState.PINCH_HOLD:
			if not is_pinching:
				# Release after hold = right-click
				self._add_event(GestureType.RIGHT_CLICK, metrics.position[:2])
				self._transition_to(GestureState.READY)
			elif is_moving:
				# Start dragging from the hold
				self.drag_start_position = metrics.position[:2]
				self._add_event(GestureType.DRAG_START, metrics.position[:2])
				self._transition_to(GestureState.DRAGGING)
		elif self.current_state == GestureState.DRAGGING:
			if not is_pinching:
				self._add_event(GestureType.DRAG_END, metrics.position[:2])
				self._transition_to(GestureState.READY)
			else:
				self._add_event(
					GestureType.DRAG_MOVE,
					metrics.position[:2],
					velocity=metrics.velocity[:2]
				)
		elif self.current_state == GestureState.DOUBLE_CLICK_WAIT:
			if time_in_state > self.double_click_window:
				self._transition_to(GestureState.READY)
			elif is_pinching:
				self._add_event(GestureType.DOUBLE_CLICK, metrics.position[:2])
				self._transition_to(GestureState.READY)

	def _handle_click(self, metrics: HandMetrics) -> None:
		"""Handle click detection with double-click support."""
		current_time = time.time()

		if (self.last_click_time and
			current_time - self.last_click_time < self.double_click_window):
			# Double click detected
			self._add_event(GestureType.DOUBLE_CLICK, metrics.position[:2])
			self.last_click_time = None
			self._transition_to(GestureState.READY)
		else:
			# Single click, wait for possible double click
			self._add_event(GestureType.CLICK, metrics.position[:2])
			self.last_click_time = current_time
			self._transition_to(GestureState.DOUBLE_CLICK_WAIT)

	def _add_hover_event(self, metrics: HandMetrics) -> None:
		"""Add hover event with smoothed position."""
		# Apply exponential smoothing
		pos = metrics.position[:2]
		if self.smoothed_position is None:
			self.smoothed_position = pos
		else:
			self.smoothed_position = (
				self.smoothing_factor * self.smoothed_position[0] + (1 - self.smoothing_factor) * pos[0],
				self.smoothing_factor * self.smoothed_position[1] + (1 - self.smoothing_factor) * pos[1]
			)
		self._add_event(
			GestureType.HOVER,
			self.smoothed_position,
			velocity=metrics.velocity[:2],
			confidence=metrics.stability_score
		)

	def _add_event(
		self,
		gesture_type: GestureType,
		position: Optional[Tuple[float, float]] = None,
		velocity: Optional[Tuple[float, float]] = None,
		confidence: float = 1.0
	) -> None:
		"""Add a gesture event to the event list."""
		event = GestureEvent(
			gesture_type=gesture_type,
			timestamp=time.time(),
			position=position,
			velocity=velocity,
			confidence=confidence
		)
		self.gesture_events.append(event)

	def _transition_to(self, new_state: GestureState) -> None:
		"""Transition to a new state."""
		if new_state != self.current_state:
			self.previous_state = self.current_state
			self.current_state = new_state
			self.state_entry_time = time.time()

	@staticmethod
	def _calculate_distance_3d(p1, p2) -> float:
		"""Calculate 3D Euclidean distance between two points."""
		dx = p1.x - p2.x
		dy = p1.y - p2.y
		dz = p1.z - p2.z
		return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

	def get_state_info(self) -> Dict:
		"""Get current state information for debugging."""
		return {
			'state': self.current_state.name,
			'time_in_state': time.time() - self.state_entry_time,
			'smoothed_position': self.smoothed_position,
			'in_activation_zone': self.current_state != GestureState.IDLE
		}

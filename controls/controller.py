import math
from dataclasses import dataclass
from typing import Tuple, Optional

# Cross-platform mouse control
try:
	import pyautogui

	pyautogui.FAILSAFE = True  # Move mouse to corner to abort
	pyautogui.PAUSE = 0.01  # Minimal delay between actions
except ImportError:
	print("Warning: pyautogui not installed. Install with: pip install pyautogui")
	pyautogui = None


@dataclass
class MouseConfig:
	"""Configuration for mouse control behavior."""
	# Screen mapping
	screen_width: int = 1920
	screen_height: int = 1080

	# Movement
	cursor_speed_multiplier: float = 1.5
	smooth_movement: bool = True
	smoothing_factor: float = 0.3

	# Click behavior
	click_duration: float = 0.05  # Seconds to hold click
	double_click_interval: float = 0.1

	# Drag behavior
	drag_smoothing: float = 0.5

	# Deadzone (center area where small movements are ignored)
	deadzone_radius: float = 0.02

	# Acceleration curve
	use_acceleration: bool = True
	acceleration_factor: float = 2.0

	# Edge margins (don't allow cursor at very edge)
	edge_margin: int = 10


class MouseController:
	"""
	Advanced mouse controller with gesture integration.

	Features:
	- Smooth cursor movement
	- Acceleration curves
	- Dead zones
	- Click, drag, and scroll support
	"""

	def __init__(self, config: Optional[MouseConfig] = None):
		if pyautogui is None:
			raise ImportError("pyautogui is required. Install with: pip install pyautogui")

		self.config = config or MouseConfig()

		# Get actual screen size
		self.screen_width, self.screen_height = pyautogui.size()
		self.config.screen_width = self.screen_width
		self.config.screen_height = self.screen_height

		# State tracking
		self.is_dragging = False
		self.last_position: Optional[Tuple[int, int]] = None
		self.smoothed_cursor_pos: Optional[Tuple[float, float]] = None

		# Calibration
		self.reference_point: Optional[Tuple[float, float]] = None
		self.calibrated = False

		print(f"Mouse Controller initialized for {self.screen_width}x{self.screen_height} screen")

	def calibrate(self, hand_position: Tuple[float, float]) -> None:
		"""
		Calibrate the controller with a reference hand position.
		This sets the center point for relative movements.
		"""
		self.reference_point = hand_position
		self.calibrated = True
		print(f"Calibrated at position: {hand_position}")

	def handle_hover(
		self,
		hand_position: Tuple[float, float],
		velocity: Optional[Tuple[float, float]] = None,
		confidence: float = 1.0
	) -> None:
		"""
		Handle hover gesture - move cursor.

		Args:
				hand_position: Normalized hand position (0-1 range)
				velocity: Movement velocity (optional, for acceleration)
				confidence: Gesture confidence (0-1)
		"""
		if not self.calibrated:
			self.calibrate(hand_position)
			return

		# Apply deadzone
		if self._in_deadzone(hand_position):
			return

		# Convert normalized coordinates to screen coordinates
		target_x, target_y = self._normalized_to_screen(hand_position)

		# Apply smoothing
		if self.config.smooth_movement:
			if self.smoothed_cursor_pos is None:
				self.smoothed_cursor_pos = (float(target_x), float(target_y))
			else:
				factor = self.config.smoothing_factor
				self.smoothed_cursor_pos = (
					factor * self.smoothed_cursor_pos[0] + (1 - factor) * target_x,
					factor * self.smoothed_cursor_pos[1] + (1 - factor) * target_y
				)
			target_x, target_y = int(self.smoothed_cursor_pos[0]), int(self.smoothed_cursor_pos[1])

		# Apply acceleration if velocity is available
		if self.config.use_acceleration and velocity:
			speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
			if speed > 0.1:  # Only accelerate on fast movements
				accel = 1.0 + (speed * self.config.acceleration_factor)
				current_x, current_y = pyautogui.position()
				dx = target_x - current_x
				dy = target_y - current_y
				target_x = int(current_x + dx * accel)
				target_y = int(current_y + dy * accel)

		# Clamp to screen bounds with margins
		target_x = max(self.config.edge_margin, min(
			self.screen_width - self.config.edge_margin, target_x
		))
		target_y = max(self.config.edge_margin, min(
			self.screen_height - self.config.edge_margin, target_y
		))

		# Move cursor
		pyautogui.moveTo(target_x, target_y)
		self.last_position = (target_x, target_y)

	def handle_click(self, position: Optional[Tuple[float, float]] = None) -> None:
		"""Handle single click gesture."""
		if position:
			x, y = self._normalized_to_screen(position)
			pyautogui.click(x, y, duration=self.config.click_duration)
		else:
			pyautogui.click(duration=self.config.click_duration)
		print("Click!")

	def handle_double_click(self, position: Optional[Tuple[float, float]] = None) -> None:
		"""Handle double click gesture."""
		if position:
			x, y = self._normalized_to_screen(position)
			pyautogui.doubleClick(x, y, interval=self.config.double_click_interval)
		else:
			pyautogui.doubleClick(interval=self.config.double_click_interval)
		print("Double click!")

	def handle_right_click(self, position: Optional[Tuple[float, float]] = None) -> None:
		"""Handle right click gesture."""
		if position:
			x, y = self._normalized_to_screen(position)
			pyautogui.rightClick(x, y)
		else:
			pyautogui.rightClick()
		print("Right click!")

	def handle_drag_start(self, position: Tuple[float, float]) -> None:
		"""Handle start of drag gesture."""
		x, y = self._normalized_to_screen(position)
		pyautogui.mouseDown(x, y)
		self.is_dragging = True
		print(f"Drag start at ({x}, {y})")

	def handle_drag_move(
		self,
		position: Tuple[float, float],
		velocity: Optional[Tuple[float, float]] = None
	) -> None:
		"""Handle drag movement."""
		if not self.is_dragging:
			return

		x, y = self._normalized_to_screen(position)

		# Apply drag smoothing
		if self.last_position:
			last_x, last_y = self.last_position
			factor = self.config.drag_smoothing
			x = int(factor * last_x + (1 - factor) * x)
			y = int(factor * last_y + (1 - factor) * y)

		pyautogui.moveTo(x, y)
		self.last_position = (x, y)

	def handle_drag_end(self, position: Tuple[float, float]) -> None:
		"""Handle end of drag gesture."""
		x, y = self._normalized_to_screen(position)
		pyautogui.mouseUp(x, y)
		self.is_dragging = False
		print(f"Drag end at ({x}, {y})")

	def handle_scroll(self, direction: str, amount: int = 1) -> None:
		"""
		Handle scroll gesture.

		Args:
				direction: 'up' or 'down'
				amount: Scroll amount (clicks)
		"""
		scroll_amount = amount if direction == 'up' else -amount
		pyautogui.scroll(scroll_amount * 120)  # 120 units per click

	def _normalized_to_screen(self, position: Tuple[float, float]) -> Tuple[int, int]:
		"""
		Convert normalized coordinates (0-1) to screen coordinates.

		Note: MediaPipe y-coordinates are inverted (0 is top)
		"""
		# Flip y-coordinate (MediaPipe has 0 at top, screen has 0 at top too)
		x = int(position[0] * self.screen_width)
		y = int(position[1] * self.screen_height)
		return x, y

	def _in_deadzone(self, position: Tuple[float, float]) -> bool:
		"""Check if position is within deadzone."""
		if not self.reference_point:
			return False

		dx = position[0] - self.reference_point[0]
		dy = position[1] - self.reference_point[1]
		distance = math.sqrt(dx ** 2 + dy ** 2)

		return distance < self.config.deadzone_radius

	def reset(self) -> None:
		"""Reset controller state."""
		if self.is_dragging:
			pyautogui.mouseUp()
			self.is_dragging = False
		self.smoothed_cursor_pos = None
		self.last_position = None
		self.calibrated = False
		self.reference_point = None


class GestureMouseIntegration:
	"""
	Integration layer between gesture recognition and mouse control.
	This is what you'll use in your main tracking loop.
	"""

	def __init__(self, mouse_config: Optional[MouseConfig] = None):
		from controls.gesture import GestureRecognizer, GestureType
		self.gesture_recognizer = GestureRecognizer()
		self.mouse_controller = MouseController(mouse_config)
		self.GestureType = GestureType

		# Statistics
		self.gesture_counts = {gt: 0 for gt in GestureType}

	def process_frame(self, landmarks: dict, frame_width: int, frame_height: int) -> None:
		"""
		Process a frame of hand landmarks and execute corresponding mouse actions.

		Args:
				landmarks: Hand landmarks from MediaPipe
				frame_width: Video frame width
				frame_height: Video frame height
		"""
		# Get gesture events from recognizer
		events = self.gesture_recognizer.update(landmarks, frame_width, frame_height)

		# Process each event
		for event in events:
			self._handle_gesture_event(event)

	def _handle_gesture_event(self, event) -> None:
		"""Handle a single gesture event."""
		gesture_type = event.gesture_type
		self.gesture_counts[gesture_type] += 1

		if gesture_type == self.GestureType.HOVER:
			self.mouse_controller.handle_hover(
				event.position,
				event.velocity,
				event.confidence
			)

		elif gesture_type == self.GestureType.CLICK:
			self.mouse_controller.handle_click(event.position)

		elif gesture_type == self.GestureType.DOUBLE_CLICK:
			self.mouse_controller.handle_double_click(event.position)

		elif gesture_type == self.GestureType.RIGHT_CLICK:
			self.mouse_controller.handle_right_click(event.position)

		elif gesture_type == self.GestureType.DRAG_START:
			self.mouse_controller.handle_drag_start(event.position)

		elif gesture_type == self.GestureType.DRAG_MOVE:
			self.mouse_controller.handle_drag_move(event.position, event.velocity)

		elif gesture_type == self.GestureType.DRAG_END:
			self.mouse_controller.handle_drag_end(event.position)

		elif gesture_type == self.GestureType.PALM_REST:
			# Hand left activation zone - optional: pause tracking
			pass

	def get_statistics(self) -> dict:
		"""Get usage statistics."""
		state_info = self.gesture_recognizer.get_state_info()
		return {
			'current_state': state_info['state'],
			'time_in_state': state_info['time_in_state'],
			'gesture_counts': self.gesture_counts,
			'is_dragging': self.mouse_controller.is_dragging,
			'calibrated': self.mouse_controller.calibrated
		}

	def reset(self) -> None:
		"""Reset both gesture recognizer and mouse controller."""
		self.mouse_controller.reset()
	# Reset gesture recognizer state if needed

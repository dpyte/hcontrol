from abc import abstractmethod, ABC
from typing import List, Dict

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.hands import Hands

from controls.location import HandLocation
from utils.constants import DEBUG, SHOW_FPS


class TrackingBase(ABC):
	""" Hand tracking base class for custom implementations """

	def __init__(
		self,
		debug: bool = DEBUG,
		show_fps: bool = SHOW_FPS,
		trace_drawing_hands: bool = False,
	):
		self.debug = debug
		self.show_fps = show_fps

		self.hand_location = HandLocation()

		self.mp_hands = mp.solutions.hands
		self.mp_drawing = mp.solutions.drawing_utils
		if trace_drawing_hands:
			self.mp_drawing_styles = mp.solutions.drawing_styles
		self.frame_count: int = 0
		self.fps: float = 0.0
		self.fps_update_interval: int = 30
		import time
		self.last_fps_update: float = time.time()

	@abstractmethod
	def run(self):
		...

	def process_frame(
		self,
		frame: np.ndarray,
		hands_processor: Hands
	) -> np.ndarray:
		"""
		Process a single frame for hand detection and tracking.

		Args:
				frame: Input BGR image frame
				hands_processor: MediaPipe Hands instance

		Returns:
				Processed image with annotations
		"""
		frame_rgb, results = self.map_hand_landmarks(frame, hands_processor)
		frame_rgb.flags.writeable = True
		frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
		# Extract and process hand landmarks if detected
		if results.multi_hand_landmarks:
			height, width = frame.shape[:2]
			for hand_landmarks in results.multi_hand_landmarks:
				landmark_data = self.extract_landmark_data(
					hand_landmarks, width, height
				)
				# using this for dev. add an option to hide this in the release
				self.hand_location.update_values(landmark_data)
				if self.debug:
					self.mp_drawing.draw_landmarks(
						frame_bgr,
						hand_landmarks,
						self.mp_hands.HAND_CONNECTIONS,
						self.mp_drawing_styles.get_default_hand_landmarks_style(),
						self.mp_drawing_styles.get_default_hand_connections_style()
					)
					angle = self.hand_location.hc_delta_theta()
					if angle > 0:
						cv2.putText(
							frame_bgr,
							f"Angle: {angle:.1f}°",
							(10, height - 40),
							cv2.FONT_HERSHEY_SIMPLEX,
							0.6,
							(0, 255, 0),
							2
						)
		if self.show_fps:
			self._update_fps()
			cv2.putText(
				frame_bgr,
				f"FPS: {self.fps:.1f}",
				(10, 30),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(0, 255, 0),
				2
			)
		return frame_bgr

	@staticmethod
	def map_hand_landmarks(frame: np.ndarray, hands_processor):
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# Ensure the array is contiguous before setting a writeable flag
		frame_rgb = np.ascontiguousarray(frame_rgb)
		frame_rgb.flags.writeable = False
		results = hands_processor.process(frame_rgb)
		return frame_rgb, results

	def get_hand_coordinates(self, results, frame, point: int):
		"""
		Returns the coordinates of a specified point from detected hand landmarks
		in a video frame. This method processes the hand landmarks detected
		via a hand-tracking module and retrieves the screen coordinates of the
		requested point.

		Parameters:
		    results: The processed output containing detected hand landmarks
		        from a detection model.
		    frame: The video frame from which the landmarks were detected.
		    point: int
		        The index of the landmark point to retrieve coordinates for.

		Returns:
		    The coordinates of the specified point as a tuple of integers if
		    found. Returns None if no hand landmarks are detected or if the
		    point's coordinates are not available.
		"""
		if not results.multi_hand_landmarks:
			return None
		height, width = frame.shape[:2]
		for hand_landmarks in results.multi_hand_landmarks:
			landmark_data = self.extract_landmark_data(hand_landmarks, width, height)
			if landmark_data is None or point > len(landmark_data):
				continue
			coords = landmark_data[point]['coordinates']
			if coords:
				return coords
		return None

	def _update_fps(self) -> None:
		"""Update FPS counter."""
		self.frame_count += 1
		if self.frame_count % self.fps_update_interval == 0:
			import time
			current_time = time.time()
			elapsed = current_time - self.last_fps_update
			self.fps = self.fps_update_interval / elapsed if elapsed > 0 else 0
			self.last_fps_update = current_time

	@staticmethod
	def extract_landmark_data(
		hand_landmarks,
		img_width: int,
		img_height: int
	) -> List[Dict]:
		"""
		Extract landmark data in the format expected by HandLocation.

		Optimized to minimize allocations and function calls.

		Args:
				hand_landmarks: MediaPipe hand landmarks
				img_width: Image width in pixels
				img_height: Image height in pixels

		Returns:
				List of landmark data dictionaries
		"""
		data = []

		# Process all 21 landmarks in a single pass
		for idx, landmark in enumerate(hand_landmarks.landmark):
			# Convert normalized coordinates to pixel coordinates
			pixel_x = min(int(landmark.x * img_width), img_width - 1)
			pixel_y = min(int(landmark.y * img_height), img_height - 1)

			# Ensure pixel coordinates are valid
			if pixel_x < 0 or pixel_y < 0:
				continue

			data.append({
				'point': idx,
				'coordinates': (pixel_x, pixel_y),
				'axis': [landmark.x, landmark.y, landmark.z]
			})
		return data

from typing import List, Tuple

import cv2
import numpy as np

from algorithm.filters import OneEuroFilter
from controls.tracking import HandTracker
from controls.tracking_base import TrackingBase
from utils.constants import FRAME_WIDTH, FRAME_HEIGHT, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, \
	MODEL_COMPLEXITY, ESC_KEY

SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440

PARENT_WINDOW_X = int((SCREEN_WIDTH - FRAME_WIDTH) / 2)
PARENT_WINDOW_Y = int((SCREEN_HEIGHT - FRAME_HEIGHT) / 2)


class Calibration(TrackingBase):
	"""
	Represents a calibration process for aligning physical and virtual points using a system of tracked hand
	gestures in a structured three-point process.

	This class is designed to handle the calibration process required for mapping real-world coordinates
	to a virtual environment using hand tracking. It provides tools to assist the user in pointing at
	specific positions on a reference plane and capturing the points with corresponding tracking data.

	Attributes:
	    SCREEN_TARGETS (List[Tuple[int, int]]): Predefined screen locations for calibration targets in
	        relative coordinates of the screen resolution.
	    CALIBRATION_TARGETS (List[Tuple[int, int]]): Predefined physical locations for calibration targets
	        based on the dimensions of the tracking frame.
	"""
	SCREEN_TARGETS: List[Tuple[int, int]] = [
		(int(SCREEN_WIDTH * 0.1), int(SCREEN_HEIGHT * 0.1)),
		(int(SCREEN_WIDTH * 0.9), int(SCREEN_HEIGHT * 0.1)),
		(int(SCREEN_WIDTH * 0.5), int(SCREEN_HEIGHT * 0.5)),
		(int(SCREEN_WIDTH * 0.1), int(SCREEN_HEIGHT * 0.9)),
		(int(SCREEN_WIDTH * 0.9), int(SCREEN_HEIGHT * 0.9))
	]
	CALIBRATION_TARGETS: List[Tuple[int, int]] = [
		(int(FRAME_WIDTH * 0.1), int(FRAME_HEIGHT * 0.1)),
		(int(FRAME_WIDTH * 0.9), int(FRAME_HEIGHT * 0.1)),
		(int(FRAME_WIDTH * 0.5), int(FRAME_HEIGHT * 0.5)),
		(int(FRAME_WIDTH * 0.1), int(FRAME_HEIGHT * 0.9)),
		(int(FRAME_WIDTH * 0.9), int(FRAME_HEIGHT * 0.9))
	]

	def __init__(self, source: int|None=0):
		super().__init__(True, True, True)
		self.all_points = []
		self.calibration_data = []
		self.current_target_index = 0
		self.filter = OneEuroFilter()
		self.is_calibrated = False
		self.physpoints = []
		self.tracker = HandTracker(False, True)
		self.transform_matrix = None

	def calibrate(self) -> bool:
		"""
		Calibrates the system using a three-point process: top-left, top-right, and center.

		The method guides the user through a step-by-step calibration process where they
		first point to specific areas and press 'c' to record the physical points. After
		collecting all required points, it creates virtual points, builds calibration
		data, computes transformation, and validates the calibration.

		After successfully validating calibration, the method saves the calibration data
		for future use. If calibration is not successful or incomplete, an appropriate
		message is printed.

		Returns:
		    bool: True if calibration was successful and validated; False otherwise.
		"""
		print("=== Calibration Started ===")
		print("Instructions:")
		print("1. Point at top-left corner, press 'c'")
		print("2. Point at top-right corner, press 'c'")
		print("3. Point at center, press 'c'")
		print("Press ESC to cancel\n")
		self.run()
		if len(self.physpoints) == 3:
			self.create_virtual_points()
			self.build_calibration_data()
			self.compute_transformation()

			if self.validate_calibration():
				print("\n[Ok] Calibration successful!")
				self.save_calibration()
				return True
			else:
				print("\n[Err] Calibration failed validation")
				return False
		else:
			print(f"\n[Err] Incomplete calibration ({len(self.physpoints)}/3 points)")
			return False

	def run(self):
		"""
		Runs the main loop for capturing video from the webcam and processes frames to calibrate hand positions
		for a specified task. This function sets up the webcam, initializes a hand tracker, and processes video
		frames to detect hand landmarks and their position. The calibration process is controlled via keyboard
		input, enabling the user to capture specific calibration points when the required hand gesture is detected.

		Raises:
		    RuntimeError: If the camera fails to open.

		"""
		cap = cv2.VideoCapture(0)
		if not cap.isOpened():
			raise RuntimeError("Failed to open camera")

		cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

		try:
			with self.tracker.mp_hands.Hands(
				min_detection_confidence=MIN_DETECTION_CONFIDENCE,
				min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
				max_num_hands=1,
				model_complexity=MODEL_COMPLEXITY,
			) as hands:
				while cap.isOpened() and self.current_target_index < 3:
					success, frame = cap.read()
					if not success:
						continue
					rgb, results = self.map_hand_landmarks(frame, hands)
					bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
					hand_position = self.get_hand_coordinates(results, frame, 8) # 8: Index
					self.draw_calibration_ui(bgr, hand_position)
					cv2.imshow('Calibration', cv2.flip(bgr, 1))
					key = cv2.waitKey(1) & 0xFF
					if hand_position is None and key == ord('c'):
						print(hand_position)
						print("DEBUG: 'c' was pressed, but no hand was detected in this frame.")
					if key == ESC_KEY:
						break
					elif key == ord('c') and hand_position:
						self.capture_calibration_point(hand_position)

		finally:
			cap.release()
			cv2.destroyAllWindows()

	def get_index_finger_position(self, results, frame) -> Tuple[int, int] | None:
		"""
		Fetches the coordinates of the index finger (landmark 8) from the given hand landmarks if available.

		This method processes detected hand landmarks to extract the position of the index finger point in
		pixel coordinates based on the provided frame dimensions. If no hands or index finger landmark is
		found in the input landmarks, the method returns None.

		Parameters:
		results (object): The detection results object containing hand landmarks.
		frame (ndarray): The frame (image) used to define the spatial resolution of the landmarks.

		Returns:
		Tuple[int, int] | None: The pixel coordinates of the index finger's position as (x, y), or None
		if the hand or index finger landmark does not exist.
		"""
		if not results.multi_hand_landmarks:
			return None
		height, width = frame.shape[:2]
		for hand_landmarks in results.multi_hand_landmarks:
			landmark_data = self.extract_landmark_data(hand_landmarks, width, height)
			coords = landmark_data[8]['coordinates']
			if coords:
				return coords
		return None

	def capture_calibration_point(self, hand_position):
		"""
		Captures a calibration point based on the provided hand position.

		This function appends the given hand position to the list of physical
		calibration points, prints a confirmation message with the current
		target index, and then increments the target index.

		Args:
		    hand_position (Any): The position of the hand to be recorded for calibration.
		"""
		self.physpoints.append(hand_position)
		print(f"[Ok] Captured point {self.current_target_index + 1}/3: {hand_position}")
		self.current_target_index += 1

	def draw_calibration_ui(self, frame, hand_position):
		"""
		Draws the calibration UI on the provided frame based on the target and the current state of
		hand position during a calibration process. The function visually indicates both the active
		target and completed targets within the calibration sequence.

		Args:
		    frame: The video frame where the calibration UI will be drawn.
		           Must be a valid image matrix compatible with OpenCV.
		    hand_position: The current position of the detected hand as a tuple of (x, y) coordinates.
		                   Can be None to indicate no hand is detected.

		Raises:
		    None
		"""
		if self.current_target_index >= 3:
			return
		target = self.CALIBRATION_TARGETS[self.current_target_index]
		cv2.circle(frame, target, 60, (0, 255, 0), 3)
		cv2.circle(frame, target, 5, (0, 255, 0), -1)
		if hand_position:
			cv2.circle(frame, hand_position, 15, (0, 0, 255), -1)
			cv2.circle(frame, hand_position, 18, (255, 255, 255), 2)

		text = f"Target {self.current_target_index + 1}/3 - Press 'c' to capture"
		cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
		for i in range(self.current_target_index):
			completed_target = self.CALIBRATION_TARGETS[i]
			cv2.circle(frame, completed_target, 60, (0, 255, 0), -1)
			cv2.putText(frame, "OK", (completed_target[0]-20, completed_target[1]+10),
									cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

	def extract_calibration_points(self, data, frame):
		"""
		Extracts calibration points from given hand landmark data and frame.

		This method processes detected hand landmarks within the provided frame. It scales
		the landmark positions relative to the frame dimensions and yields the corresponding
		landmark data. If no valid landmark data is extracted, it will yield a fallback
		value of None.

		Yields:
		    dict | None: Extracted landmark data as a dictionary or None if no valid
		    data is available or after processing final cleanup.

		Arguments:
		    data: The object containing multiple hand landmarks detected in the frame.
		    frame: The image frame from which landmarks are detected. It is expected
		    to have a defined shape with height and width.
		"""
		height, width = frame.shape[:2]
		try:
			for hl in data.multi_hand_landmarks:
				yield self.extract_landmark_data(hl, width, height)
		finally:
			yield None

	def smooth_points(self, points) -> Tuple[float, float]:
		"""
		Smooths the provided coordinates using a smoothing filter.

		This method takes a tuple of coordinates from the input dictionary and applies
		a smoothing function using an external filter object. It is useful for adjusting
		or refining data points for further processing, ensuring the coordinates are smooth.

		Parameters:
		    points: dict
		        A dictionary containing 'coordinates' as a key with a tuple of x and y
		        coordinate values.

		Returns:
		    Tuple[float, float]: A tuple containing the smoothed x and y coordinate values.
		"""
		(x, y) = points['coordinates']
		return self.filter.smooth(x, y)

	def create_virtual_points(self):
		P1_x, P1_y = self.physpoints[0]
		P2_x, P2_y = self.physpoints[1]
		V1 = (P1_x, FRAME_HEIGHT - P1_y)
		V2 = (P2_x, FRAME_HEIGHT - P2_y)
		self.all_points = self.physpoints + [V1, V2]
		print(f"\nPhysical points captured:")
		print(f"  P1 (top-left):  {self.physpoints[0]}")
		print(f"  P2 (top-right): {self.physpoints[1]}")
		print(f"  P3 (center):    {self.physpoints[2]}")
		print(f"\nVirtual points created:")
		print(f"  V1 (bottom-left):  {V1}")
		print(f"  V2 (bottom-right): {V2}")

	def build_calibration_data(self):
		self.calibration_data = [
			{'hand': self.all_points[i], 'screen': self.SCREEN_TARGETS[i]}
			for i in range(5)
		]

	def compute_transformation(self):
		hand_points = np.array([p['hand'] for p in self.calibration_data[:4]],
													 dtype=np.float32)
		screen_points = np.array([p['screen'] for p in self.calibration_data[:4]],
														 dtype=np.float32)
		self.transform_matrix = cv2.getPerspectiveTransform(hand_points, screen_points)
		self.is_calibrated = True
		print("\n[Ok] Transformation matrix computed")

	def validate_calibration(self) -> bool:
		x_coords = [p[0] for p in self.physpoints]
		x_range = max(x_coords) - min(x_coords)
		if x_range < FRAME_WIDTH * 0.5:
			print(f"[Warn] Warning: Horizontal spread is small ({x_range:.0f}px)")
			return False

		y_coords = [p[1] for p in self.physpoints]
		y_range = max(y_coords) - min(y_coords)
		if y_range < FRAME_HEIGHT * 0.3:
			print(f"[Warn] Warning: Vertical spread is small ({y_range:.0f}px)")
			return False
		print(f"[Ok] Point spread: {x_range:.0f}px x {y_range:.0f}px")
		return True

	def map_hand_to_screen(self, hand_x: float, hand_y: float) -> Tuple[int, int]:
		if not self.is_calibrated:
			raise RuntimeError("Not calibrated!")
		hand_point = np.array([[[hand_x, hand_y]]], dtype=np.float32)
		screen_point = cv2.perspectiveTransform(hand_point, self.transform_matrix)
		screen_x = int(screen_point[0, 0, 0])
		screen_y = int(screen_point[0, 0, 1])
		screen_x = max(0, min(screen_x, SCREEN_WIDTH - 1))
		screen_y = max(0, min(screen_y, SCREEN_HEIGHT - 1))
		return screen_x, screen_y

	def save_calibration(self, filepath='calibration.npz'):
		import time
		np.savez(filepath,
						 transform_matrix=self.transform_matrix,
						 calibration_data=self.calibration_data,
						 screen_width=SCREEN_WIDTH,
						 screen_height=SCREEN_HEIGHT,
						 timestamp=time.time())
		print(f"Calibration saved to {filepath}")

	def load_calibration(self, filepath='calibration.npz') -> bool:
		try:
			data = np.load(filepath, allow_pickle=True)
			self.transform_matrix = data['transform_matrix']
			self.is_calibrated = True
			print(f"[Ok] Calibration loaded from {filepath}")
			return True
		except:
			print(f"[Err] Failed to load calibration from {filepath}")
			return False

	@staticmethod
	def show_window(processed_frame):
		window_name: str = 'Calibration window'
		cv2.imshow(window_name, cv2.flip(processed_frame, 1))
		cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

	@staticmethod
	def distance_to_target(hand_pos, target):
		return np.linalg.norm(np.array(hand_pos) - np.array(target))

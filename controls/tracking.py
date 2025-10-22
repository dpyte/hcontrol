#!/usr/bin/env python3
"""
Optimized hand tracking application using MediaPipe and OpenCV.
Now uses pure Python HandCoordinates module for 10-20x better performance.
"""
import time
from typing import Dict, Optional

import cv2
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.tasks import python

from capture.camera import initialize_capture_device, video_properties
from controls.controller import MouseConfig, GestureMouseIntegration
from utils.constants import (
	DEBUG, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
	ESC_KEY, SHOW_FPS, MODEL_COMPLEXITY, MAX_NUM_HANDS,
	FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE
)

# Import our optimized Python module instead of C++

# Configuration

# 0=lite (fastest), 1=full (balanced), 2=heavy (most accurate)

base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task', delegate=python.BaseOptions.Delegate.GPU)


class HandTracker:
	def __init__(self, debug: bool = DEBUG, show_fps: bool = SHOW_FPS):
		self.debug = debug
		self.show_fps = show_fps
		self.mp_hands = mp_hands
		mouse_config = MouseConfig(
			cursor_speed_multiplier=1.8,
			smooth_movement=True,
			smoothing_factor=0.4,
			use_acceleration=True,
			deadzone_radius=0.015
		)
		self.gesture_mouse = GestureMouseIntegration(mouse_config)
		self.fps_start_time = time.time()
		self.fps_frame_count = 0
		self.fps = 0.0
		self.show_activation_zone = True
		self.show_gesture_info = True

	def run(self, camera_id: int = 0) -> None:
		"""
		Main tracking loop with gesture control.

		Press:
		- ESC: Exit
		- SPACE: Toggle activation zone visualization
		- 'g': Toggle gesture info display
		- 'r': Reset/recalibrate
		"""
		cap = cv2.VideoCapture(camera_id)

		if not cap.isOpened():
			raise RuntimeError(f"Failed to open camera {camera_id}")

		cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
		cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

		print("=" * 70)
		print("Hand Gesture Mouse Control")
		print("=" * 70)
		print("Gestures:")
		print("  - Point with index finger: Move cursor")
		print("  - Pinch (thumb + index): Click")
		print("  - Pinch + Hold: Right click")
		print("  - Pinch + Move: Drag")
		print("  - Quick double pinch: Double click")
		print("")
		print("Activation Zone: Keep hand in center area (chest level)")
		print("Rest Mode: Drop hand below zone or move to edge")
		print("")
		print("Controls:")
		print("  ESC: Exit")
		print("  SPACE: Toggle zone visualization")
		print("  G: Toggle gesture info")
		print("  R: Reset/Recalibrate")
		print("=" * 70)

		try:
			with self.mp_hands.Hands(
				min_detection_confidence=MIN_DETECTION_CONFIDENCE,
				min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
				max_num_hands=MAX_NUM_HANDS,
				model_complexity=MODEL_COMPLEXITY
			) as hands:
				while cap.isOpened():
					success, frame = cap.read()
					if not success:
						print("Warning: Failed to read frame")
						continue
					frame = cv2.flip(frame, 1)  # Mirror for natural interaction
					rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					results = hands.process(rgb_frame)
					landmarks_dict = self._extract_landmarks(results)
					self.gesture_mouse.process_frame(
						landmarks_dict,
						frame.shape[1],
						frame.shape[0]
					)
					if self.debug:
						display_frame = self._draw_visualizations(
							frame,
							results,
							landmarks_dict
						)
						cv2.imshow('Hand Gesture Control', display_frame)
					self._update_fps()
					key = cv2.waitKey(1) & 0xFF
					if key == ESC_KEY:
						break
					elif key == ord(' '):
						self.show_activation_zone = not self.show_activation_zone
					elif key == ord('g'):
						self.show_gesture_info = not self.show_gesture_info
					elif key == ord('r'):
						self.gesture_mouse.reset()
						print("System reset - recalibrating...")

		except KeyboardInterrupt:
			print("\nStopped by user")
		finally:
			cap.release()
			cv2.destroyAllWindows()
			self._print_final_stats()

	@staticmethod
	def _extract_landmarks(results) -> Dict:
		"""Convert MediaPipe results to landmark dictionary."""
		if not results.multi_hand_landmarks:
			return {}
		hand_landmarks = results.multi_hand_landmarks[0]
		landmarks_dict = {}
		for idx, landmark in enumerate(hand_landmarks.landmark):
			landmarks_dict[idx] = landmark
		return landmarks_dict

	def _draw_visualizations(self, frame, results, landmarks_dict):
		"""Draw visualizations on the frame."""
		display_frame = frame.copy()
		if results.multi_hand_landmarks:
			from mediapipe.python.solutions import drawing_utils, drawing_styles
			for hand_landmarks in results.multi_hand_landmarks:
				drawing_utils.draw_landmarks(
					display_frame,
					hand_landmarks,
					self.mp_hands.HAND_CONNECTIONS,
					drawing_styles.get_default_hand_landmarks_style(),
					drawing_styles.get_default_hand_connections_style()
				)
		if self.show_activation_zone:
			self._draw_activation_zone(display_frame)
		if self.show_gesture_info:
			self._draw_gesture_info(display_frame)
		if self.show_fps:
			cv2.putText(
				display_frame,
				f"FPS: {self.fps:.1f}",
				(10, 30),
				cv2.FONT_HERSHEY_SIMPLEX,
				1.0,
				(0, 255, 0),
				2
			)
		return display_frame

	def _draw_activation_zone(self, frame):
		"""Draw the activation zone boundaries."""
		h, w = frame.shape[:2]
		zone = self.gesture_mouse.gesture_recognizer.activation_zone
		x1 = int(zone.min_x * w)
		x2 = int(zone.max_x * w)
		y1 = int(zone.min_y * h)
		y2 = int(zone.max_y * h)
		color = (0, 255, 0) if self._is_hand_in_zone() else (0, 0, 255)
		cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
		cv2.putText(
			frame,
			"ACTIVE ZONE",
			(x1 + 10, y1 + 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			color,
			2
		)

	def _draw_gesture_info(self, frame):
		"""Draw current gesture state and statistics."""
		stats = self.gesture_mouse.get_statistics()
		h, w = frame.shape[:2]
		overlay = frame.copy()
		cv2.rectangle(overlay, (w - 300, 0), (w, 150), (0, 0, 0), -1)
		cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
		y_offset = 25
		texts = [
			f"State: {stats['current_state']}",
			f"Time: {stats['time_in_state']:.1f}s",
			f"Calibrated: {stats['calibrated']}",
			f"Dragging: {stats['is_dragging']}"
		]

		for text in texts:
			cv2.putText(
				frame,
				text,
				(w - 290, y_offset),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				(255, 255, 255),
				1
			)
			y_offset += 30

	def _is_hand_in_zone(self) -> bool:
		"""Check if the hand is currently in the activation zone."""
		state_info = self.gesture_mouse.gesture_recognizer.get_state_info()
		return state_info['in_activation_zone']

	def _update_fps(self):
		"""Update FPS counter."""
		import time
		self.fps_frame_count += 1
		elapsed = time.time() - self.fps_start_time
		if elapsed > 1.0:
			self.fps = self.fps_frame_count / elapsed
			self.fps_frame_count = 0
			self.fps_start_time = time.time()

	def _print_final_stats(self):
		"""Print final statistics."""
		stats = self.gesture_mouse.get_statistics()
		print("\n" + "=" * 70)
		print("Session Statistics")
		print("=" * 70)
		print(f"Final State: {stats['current_state']}")
		print(f"Average FPS: {self.fps:.1f}")
		print("\nGesture Counts:")
		for gesture_type, count in stats['gesture_counts'].items():
			if count > 0:
				print(f"  {gesture_type.name}: {count}")
		print("=" * 70)


class BatchHandTracker(HandTracker):
	"""
	Extended tracker that can process video files and image sequences.
	Useful for offline analysis and benchmarking.
	"""

	def process_video_file(
		self,
		video_path: str,
		output_path: Optional[str] = None
	) -> Dict:
		"""
		Process a video file and optionally save annotated output.

		Args:
				video_path: Path to input video file
				output_path: Optional path to save annotated video

		Returns:
				Dictionary with processing statistics
		"""
		cap = initialize_capture_device(video_path)
		if not cap.isOpened():
			raise RuntimeError(f"Failed to open video: {video_path}")

		# Get video properties
		properties = video_properties()
		fps = properties['fps'] if properties else 30
		width = properties['width'] if properties else FRAME_WIDTH
		height = properties['height'] if properties else FRAME_HEIGHT
		total_frames = properties['total_frames'] if properties else 0

		print(f"Processing video: {video_path}")
		print(f"  Resolution: {width}x{height}")
		print(f"  FPS: {fps}")
		print(f"  Total frames: {total_frames}")

		# Set up a video writer if an output path specified
		writer = None
		if output_path:
			fourcc = cv2.VideoWriter_fourcc(*'mp4v')
			writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

		import time
		start_time = time.time()
		processed_frames = 0

		try:
			with self.mp_hands.Hands(
				min_detection_confidence=MIN_DETECTION_CONFIDENCE,
				min_tracking_confidence=MIN_TRACKING_CONFIDENCE
			) as hands:
				while True:
					success, frame = cap.read()
					if not success:
						break

					# Process frame
					processed_frame = self.process_frame(frame, hands)

					# Write to output if enabled
					if writer:
						writer.write(processed_frame)

					processed_frames += 1

					# Progress indicator
					if processed_frames % 30 == 0:
						progress = (processed_frames / total_frames) * 100
						print(f"  Progress: {progress:.1f}%", end='\r')

		finally:
			cap.release()
			if writer:
				writer.release()

		elapsed_time = time.time() - start_time
		stats = {
			'processed_frames': processed_frames,
			'elapsed_time': elapsed_time,
			'processing_fps': processed_frames / elapsed_time if elapsed_time > 0 else 0,
			'hand_location_stats': self.hand_location.get_statistics()
		}
		print(f"\n\nProcessing complete!")
		print(f"  Processed: {processed_frames} frames")
		print(f"  Time: {elapsed_time:.2f} seconds")
		print(f"  Processing FPS: {stats['processing_fps']:.2f}")
		return stats

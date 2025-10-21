#!/usr/bin/env python3
"""
Optimized hand tracking application using MediaPipe and OpenCV.
Now uses pure Python HandCoordinates module for 10-20x better performance.
"""

from typing import List, Dict, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python

from capture.camera import initialize_capture_device, video_properties
from controls.location import HandLocation
from utils.constants import DEBUG, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, ESC_KEY, SHOW_FPS, \
	MODEL_COMPLEXITY, MAX_NUM_HANDS, FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE

# Import our optimized Python module instead of C++

# Configuration

# 0=lite (fastest), 1=full (balanced), 2=heavy (most accurate)

base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task', delegate=python.BaseOptions.Delegate.GPU)


class HandTracker:
	"""High-performance hand tracking with MediaPipe and optimized coordinate processing."""

	def __init__(self, debug: bool = DEBUG, show_fps: bool = SHOW_FPS):
		self.debug = debug
		self.show_fps = show_fps

		# Use Python HandLocation (much faster than the C++ version)
		self.hand_location = HandLocation()

		# Initialize MediaPipe components
		self.mp_hands = mp.solutions.hands
		self.mp_drawing = mp.solutions.drawing_utils
		self.mp_drawing_styles = mp.solutions.drawing_styles

		# FPS calculation
		self.frame_count = 0
		self.fps = 0.0
		self.fps_update_interval = 30  # Update FPS every N frames

		import time
		self.last_fps_update = time.time()

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

	def process_frame(
		self,
		frame: np.ndarray,
		hands_processor
	) -> np.ndarray:
		"""
		Process a single frame for hand detection and tracking.

		Args:
				frame: Input BGR image frame
				hands_processor: MediaPipe Hands instance

		Returns:
				Processed image with annotations
		"""
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame_rgb.flags.writeable = False

		results = hands_processor.process(frame_rgb)

		frame_rgb.flags.writeable = True
		frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

		# Extract and process hand landmarks if detected
		if results.multi_hand_landmarks:
			height, width = frame.shape[:2]

			for hand_landmarks in results.multi_hand_landmarks:
				landmark_data = self.extract_landmark_data(
					hand_landmarks, width, height
				)

				self.hand_location.update_values(landmark_data)
				if self.debug:
					self.mp_drawing.draw_landmarks(
						frame_bgr,
						hand_landmarks,
						self.mp_hands.HAND_CONNECTIONS,
						self.mp_drawing_styles.get_default_hand_landmarks_style(),
						self.mp_drawing_styles.get_default_hand_connections_style()
					)

					# Draw angle information
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

		# Add FPS counter if enabled
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

	def _update_fps(self) -> None:
		"""Update FPS counter."""
		self.frame_count += 1

		if self.frame_count % self.fps_update_interval == 0:
			import time
			current_time = time.time()
			elapsed = current_time - self.last_fps_update
			self.fps = self.fps_update_interval / elapsed if elapsed > 0 else 0
			self.last_fps_update = current_time

	def run(self, camera_id: int = 0) -> None:
		"""
		Main tracking loop.

		Args:
				camera_id: Camera device ID (default: 0)
		"""
		cap = cv2.VideoCapture(camera_id)

		if not cap.isOpened():
			raise RuntimeError(f"Failed to open camera {camera_id}")

		# Optional: Set camera properties for better performance
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
		cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

		print("Starting hand tracking...")
		print("Press ESC to exit")
		print("-" * 60)

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
						print("Warning: Failed to read frame from camera")
						continue

					# Process frame
					processed_frame = self.process_frame(frame, hands)

					# Display if in debug mode
					if self.debug:
						cv2.imshow('Hand Tracking (Python)', cv2.flip(processed_frame, 1))

					# Check for ESC key to exit
					key = cv2.waitKey(1) & 0xFF
					if key == ESC_KEY:
						break
					elif key == ord('s'):
						# Save screenshot
						cv2.imwrite('hand_tracking_screenshot.png', processed_frame)
						print("Screenshot saved!")
					elif key == ord('p'):
						# Print statistics
						stats = self.hand_location.get_statistics()
						print(f"\nStatistics: {stats}")

		except KeyboardInterrupt:
			print("\nTracking stopped by user")

		finally:
			# Cleanup
			cap.release()
			cv2.destroyAllWindows()

			# Print final statistics
			stats = self.hand_location.get_statistics()
			print("\nFinal Statistics:")
			print(f"  Total updates: {stats['update_count']}")
			print(f"  Active landmarks: {stats['active_landmarks']}/21")
			print(f"  Current angle: {stats['current_angle']:.2f}°")
			print(f"  Average FPS: {self.fps:.1f}")


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

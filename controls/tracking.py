#!/usr/bin/env python3
"""
Optimized hand tracking application using MediaPipe and OpenCV.
Now uses pure Python HandCoordinates module for 10-20x better performance.
"""
from typing import Dict, Optional, Any, Generator

import cv2
from mediapipe.tasks import python

from capture.camera import initialize_capture_device, video_properties
from controls.tracking_base import TrackingBase
from controls.types import ExtractionPoints
from utils.constants import DEBUG, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, ESC_KEY, SHOW_FPS, \
	MODEL_COMPLEXITY, MAX_NUM_HANDS, FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE

# Import our optimized Python module instead of C++

# Configuration

# 0=lite (fastest), 1=full (balanced), 2=heavy (most accurate)

base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task', delegate=python.BaseOptions.Delegate.GPU)


class HandTracker(TrackingBase):
	"""High-performance hand tracking with MediaPipe and optimized coordinate processing."""

	def __init__(self, debug: bool = DEBUG, show_fps: bool = SHOW_FPS):
		super().__init__(debug, show_fps, trace_drawing_hands=True)


	def run(self, camera_id: int = 0) -> None:
		"""
		Main tracking loop.

		Args:
				camera_id: Camera device ID (default: 0)
		"""
		cap = cv2.VideoCapture(camera_id)

		if not cap.isOpened():
			raise RuntimeError(f"Failed to open camera {camera_id}")

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
					rgb, results = self.map_hand_landmarks(frame, hands)
					processed_frame = self.process_frame(frame, hands)
					points = self.extract_key_points(results, frame)

					if self.debug:
						cv2.imshow('Hand Tracking', cv2.flip(processed_frame, 1))
					key = cv2.waitKey(1) & 0xFF
					if key == ESC_KEY:
						break
					elif key == ord('p'):
						stats = self.hand_location.get_statistics()
						print(f"\nStatistics: {stats}")
		except KeyboardInterrupt:
			print("\nTracking stopped by user")
		finally:
			cap.release()
			cv2.destroyAllWindows()
			stats = self.hand_location.get_statistics()
			print("\nFinal Statistics:")
			print(f"  Total updates: {stats['update_count']}")
			print(f"  Active landmarks: {stats['active_landmarks']}/21")
			print(f"  Current angle: {stats['current_angle']:.2f}°")
			print(f"  Average FPS: {self.fps:.1f}")

	def extract_key_points(self, results, frame):
		palm_pos  = self.get_hand_coordinates(results, frame, 0)
		thumb_pos = self.get_hand_coordinates(results, frame, 4)
		index_pos = self.get_hand_coordinates(results, frame, 8)
		middle_pos = self.get_hand_coordinates(results, frame, 12)
		ring_pos = self.get_hand_coordinates(results, frame, 16)
		pinky_pos = self.get_hand_coordinates(results, frame, 20)
		return ExtractionPoints(palm_pos, thumb_pos, index_pos, middle_pos, ring_pos, pinky_pos)


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

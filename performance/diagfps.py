#!/usr/bin/env python3
"""
Performance diagnostic tool for hand tracking.
Identifies bottlenecks and suggests optimizations.
"""

import time
from typing import Dict

import cv2
import mediapipe as mp
import numpy as np


def diagnose_camera(camera_id: int = 0) -> Dict:
	"""Test camera I/O performance."""
	print("\n" + "=" * 60)
	print("DIAGNOSING CAMERA I/O")
	print("=" * 60)

	cap = cv2.VideoCapture(camera_id)
	if not cap.isOpened():
		print("❌ Failed to open camera")
		return {}

	# Get camera properties
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS)

	print(f"Camera resolution: {width}×{height}")
	print(f"Camera FPS setting: {fps}")

	# Test read speed
	times = []
	print("\nTesting camera read speed (30 frames)...")
	for i in range(30):
		start = time.perf_counter()
		ret, frame = cap.read()
		elapsed = (time.perf_counter() - start) * 1000
		times.append(elapsed)
		if not ret:
			print(f"❌ Failed to read frame {i}")
			break

	cap.release()

	avg_time = np.mean(times)
	max_fps = 1000 / avg_time if avg_time > 0 else 0

	print(f"\nCamera read time: {avg_time:.2f}ms per frame")
	print(f"Maximum camera FPS: {max_fps:.1f}")

	if avg_time > 20:
		print("⚠️  WARNING: Camera I/O is slow!")
		print("   Solution: Try a different camera or lower resolution")
	else:
		print("✓ Camera I/O is fast enough")

	return {
		'width': width,
		'height': height,
		'avg_read_time_ms': avg_time,
		'max_fps': max_fps
	}


def diagnose_mediapipe(complexity: int = 1, resolution: tuple = (640, 480)) -> Dict:
	"""Test MediaPipe processing speed."""
	print("\n" + "=" * 60)
	print(f"DIAGNOSING MEDIAPIPE (complexity={complexity})")
	print("=" * 60)

	mp_hands = mp.solutions.hands

	# Create dummy frames
	dummy_frame = np.random.randint(0, 255, (*resolution[::-1], 3), dtype=np.uint8)

	times = []
	print(f"\nTesting MediaPipe processing (30 frames at {resolution[0]}×{resolution[1]})...")

	with mp_hands.Hands(
		min_detection_confidence=0.9,
		min_tracking_confidence=0.5,
		model_complexity=complexity,
		max_num_hands=2
	) as hands:
		# Warm-up
		for _ in range(5):
			_ = hands.process(cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB))

		# Actual test
		for i in range(30):
			rgb_frame = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)
			start = time.perf_counter()
			results = hands.process(rgb_frame)
			elapsed = (time.perf_counter() - start) * 1000
			times.append(elapsed)

	avg_time = np.mean(times)
	max_fps = 1000 / avg_time if avg_time > 0 else 0

	print(f"\nMediaPipe processing: {avg_time:.2f}ms per frame")
	print(f"Maximum FPS: {max_fps:.1f}")

	if avg_time > 30:
		print("⚠️  WARNING: MediaPipe processing is slow!")
		print("   Solutions:")
		print("   1. Set MODEL_COMPLEXITY = 0")
		print("   2. Lower camera resolution")
		print("   3. Install mediapipe-gpu")
	elif avg_time > 15:
		print("⚠️  MediaPipe is moderate speed")
		print("   Try MODEL_COMPLEXITY = 0 for better performance")
	else:
		print("✓ MediaPipe processing is fast")

	return {
		'avg_processing_time_ms': avg_time,
		'max_fps': max_fps,
		'complexity': complexity
	}


def diagnose_coordinate_processing() -> Dict:
	"""Test coordinate processing speed."""
	print("\n" + "=" * 60)
	print("DIAGNOSING COORDINATE PROCESSING")
	print("=" * 60)

	try:
		from controls.location import HandLocation
		hand_loc = HandLocation()
	except ImportError:
		print("⚠️  Could not import HandLocation from controls.location")
		print("   Using benchmark module instead...")
		try:
			from hand_coordinates import HandLocation
			hand_loc = HandLocation()
		except ImportError:
			print("❌ Could not import HandLocation")
			return {}

	# Generate test data
	test_data = [
		{'point': i, 'coordinates': (100 + i * 10, 200 + i * 10),
		 'axis': [0.1 + i * 0.02, 0.2 + i * 0.02, 0.3 + i * 0.02]}
		for i in range(21)
	]

	times = []
	print("\nTesting coordinate processing (1000 updates)...")

	for _ in range(1000):
		start = time.perf_counter()
		hand_loc.update_values(test_data)
		elapsed = (time.perf_counter() - start) * 1e6  # microseconds
		times.append(elapsed)

	avg_time = np.mean(times)

	print(f"\nCoordinate processing: {avg_time:.2f}μs per frame")
	print(f"Percentage of 30 FPS budget: {(avg_time / 33333) * 100:.2f}%")

	if avg_time > 1000:
		print("⚠️  WARNING: Coordinate processing is slow!")
		print("   This should be < 100μs. Check your implementation.")
	else:
		print("✓ Coordinate processing is fast (not a bottleneck)")

	return {
		'avg_time_us': avg_time,
		'pct_of_frame_budget': (avg_time / 33333) * 100
	}


def diagnose_visualization() -> Dict:
	"""Test visualization overhead."""
	print("\n" + "=" * 60)
	print("DIAGNOSING VISUALIZATION")
	print("=" * 60)

	mp_hands = mp.solutions.hands
	mp_drawing = mp.solutions.drawing_utils
	mp_drawing_styles = mp.solutions.drawing_styles

	# Create dummy frame with landmarks
	dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

	# Create dummy hand landmarks
	with mp_hands.Hands(model_complexity=0) as hands:
		rgb_frame = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)
		results = hands.process(rgb_frame)

		if results.multi_hand_landmarks:
			hand_landmarks = results.multi_hand_landmarks[0]

			times = []
			print("\nTesting visualization drawing (100 frames)...")

			for _ in range(100):
				test_frame = dummy_frame.copy()
				start = time.perf_counter()
				mp_drawing.draw_landmarks(
					test_frame,
					hand_landmarks,
					mp_hands.HAND_CONNECTIONS,
					mp_drawing_styles.get_default_hand_landmarks_style(),
					mp_drawing_styles.get_default_hand_connections_style()
				)
				elapsed = (time.perf_counter() - start) * 1000
				times.append(elapsed)

			avg_time = np.mean(times)
			print(f"\nVisualization drawing: {avg_time:.2f}ms per frame")

			if avg_time > 5:
				print("⚠️  Visualization overhead is high")
				print("   Solution: Set DEBUG = False for better performance")
			else:
				print("✓ Visualization overhead is acceptable")

			return {'avg_time_ms': avg_time}

	print("⚠️  Could not test (no hand detected in dummy frame)")
	return {}


def run_full_diagnostic():
	"""Run complete performance diagnostic."""
	print("=" * 60)
	print("HAND TRACKING PERFORMANCE DIAGNOSTIC")
	print("=" * 60)

	results = {}

	# 1. Camera I/O
	results['camera'] = diagnose_camera()

	# 2. MediaPipe at different complexities
	if results['camera']:
		resolution = (results['camera']['width'], results['camera']['height'])
	else:
		resolution = (640, 480)

	print("\n--- Testing MODEL_COMPLEXITY = 0 (Lite) ---")
	results['mediapipe_lite'] = diagnose_mediapipe(0, resolution)

	print("\n--- Testing MODEL_COMPLEXITY = 1 (Full) ---")
	results['mediapipe_full'] = diagnose_mediapipe(1, resolution)

	print("\n--- Testing MODEL_COMPLEXITY = 2 (Heavy) ---")
	results['mediapipe_heavy'] = diagnose_mediapipe(2, resolution)

	# 3. Coordinate processing
	results['coordinates'] = diagnose_coordinate_processing()

	# 4. Visualization
	results['visualization'] = diagnose_visualization()

	# Summary and recommendations
	print("\n" + "=" * 60)
	print("PERFORMANCE SUMMARY & RECOMMENDATIONS")
	print("=" * 60)

	# Calculate total frame time estimate
	if all(key in results for key in ['camera', 'mediapipe_full', 'coordinates']):
		camera_time = results['camera'].get('avg_read_time_ms', 0)
		mediapipe_time = results['mediapipe_full'].get('avg_processing_time_ms', 0)
		coord_time = results['coordinates'].get('avg_time_us', 0) / 1000
		viz_time = results.get('visualization', {}).get('avg_time_ms', 0)

		total_time = camera_time + mediapipe_time + coord_time + viz_time
		estimated_fps = 1000 / total_time if total_time > 0 else 0

		print(f"\nEstimated frame breakdown (MODEL_COMPLEXITY=1):")
		print(f"  Camera I/O:        {camera_time:.1f}ms ({camera_time / total_time * 100:.0f}%)")
		print(f"  MediaPipe:         {mediapipe_time:.1f}ms ({mediapipe_time / total_time * 100:.0f}%)")
		print(f"  Coordinates:       {coord_time:.1f}ms ({coord_time / total_time * 100:.0f}%)")
		print(f"  Visualization:     {viz_time:.1f}ms ({viz_time / total_time * 100:.0f}%)")
		print(f"  ─────────────────────────────")
		print(f"  Total:             {total_time:.1f}ms")
		print(f"  Estimated FPS:     {estimated_fps:.1f}")

		print("\n🎯 RECOMMENDATIONS:")

		if estimated_fps < 20:
			print("\n❌ Your system is slow. Apply these fixes:")

			# Check MediaPipe
			lite_time = results['mediapipe_lite'].get('avg_processing_time_ms', 0)
			full_time = results['mediapipe_full'].get('avg_processing_time_ms', 0)

			if full_time > 25:
				speedup = full_time / lite_time if lite_time > 0 else 1
				print(f"\n1. SET MODEL_COMPLEXITY = 0")
				print(f"   Current: {full_time:.1f}ms → With lite: {lite_time:.1f}ms")
				print(f"   Expected FPS gain: {estimated_fps:.1f} → {1000 / (total_time - full_time + lite_time):.1f}")

			if camera_time > 20:
				print(f"\n2. LOWER CAMERA RESOLUTION")
				print(f"   Current: {results['camera']['width']}×{results['camera']['height']}")
				print(f"   Try: 320×240 or 640×480")

			if viz_time > 3:
				print(f"\n3. DISABLE DEBUG VISUALIZATION")
				print(f"   Set DEBUG = False")
				print(f"   Save: {viz_time:.1f}ms per frame")

			print(f"\n4. LOWER CONFIDENCE THRESHOLDS")
			print(f"   MIN_DETECTION_CONFIDENCE = 0.7")
			print(f"   MIN_TRACKING_CONFIDENCE = 0.3")

			print(f"\n5. TRACK ONE HAND ONLY")
			print(f"   MAX_NUM_HANDS = 1")

		elif estimated_fps < 25:
			print("\n⚠️  Performance is moderate. Try:")
			print("1. MODEL_COMPLEXITY = 0 for better FPS")
			print("2. Consider installing mediapipe-gpu")

		else:
			print("\n✓ Your system is fast enough!")
			print("  If actual FPS is lower, check:")
			print("  - Is DEBUG = True? (adds visualization overhead)")
			print("  - Are other programs using CPU/camera?")

	print("\n" + "=" * 60)


if __name__ == '__main__':
	try:
		run_full_diagnostic()
	except KeyboardInterrupt:
		print("\n\nDiagnostic interrupted by user")
	except Exception as e:
		print(f"\n\nError during diagnostic: {e}")
		import traceback

		traceback.print_exc()

#!/usr/bin/env python3
"""Hand tracking application using MediaPipe and OpenCV."""
from calibration.logic import Calibration
from controls.tracking import HandTracker, BatchHandTracker

DEBUG = True


def main():
	"""Entry point for the hand tracking application."""
	import argparse

	parser = argparse.ArgumentParser(description='Hand Tracking with Optimized Python Backend')
	parser.add_argument('--run-calibration', action='store_true', help='Update calibration data')
	parser.add_argument('--no-debug', action='store_true', help='Disable debug visualization')
	parser.add_argument('--no-fps', action='store_true', help='Disable FPS counter')
	parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
	parser.add_argument('--video', type=str, help='Process video file instead of camera')
	parser.add_argument('--output', type=str, help='Output video path (with --video)')
	parser.add_argument('--write-coords', action='store_true', help='Write coordinates to file')

	args = parser.parse_args()

	try:
		if args.run_calibration:
			# No need to run further from here on
			calibration = Calibration(0)
			calibration.calibrate()
			return None

		if args.video:
			# Batch processing mode
			tracker = BatchHandTracker(debug=not args.no_debug, show_fps=not args.no_fps)
			if args.write_coords:
				tracker.hand_location.enable_coordinates_write_out()
			tracker.process_video_file(args.video, args.output)
		else:
			# Real-time camera mode
			tracker = HandTracker(debug=not args.no_debug, show_fps=not args.no_fps)
			if args.write_coords:
				tracker.hand_location.enable_coordinates_write_out()
			tracker.run(camera_id=args.camera)

	except Exception as e:
		print(f"Error: {e}")
		import traceback
		traceback.print_exc()
		return 1

	return 0


if __name__ == '__main__':
	exit(main())

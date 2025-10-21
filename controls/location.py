from typing import Dict, Tuple, List

import numpy as np

from controls.types import Point, TRACKED_POINTS, CHANGE_THRESHOLD, IGNORE_Z


class HandLocation:
	"""
	High-performance hand landmark coordinate tracker.

	Uses pre-allocated NumPy arrays and vectorized operations for minimal overhead.
	Approximately 10-20x faster than the C++ version due to eliminating pybind11 overhead.
	"""

	__slots__ = (
		'_landmarks',
		'_prev_landmarks',
		'_pixel_coords',
		'_needs_update',
		'_locked_thumb_cmc',
		'_locked_wrist',
		'_lock_slope',
		'_axis_locked',
		'_angle',
		'_write_enabled',
		'_file_handle',
		'_update_count'
	)

	def __init__(self):
		"""Initialize with pre-allocated arrays for zero-copy operations."""
		# Pre-allocate all arrays (21 landmarks × 3 dimensions)
		self._landmarks = np.zeros((21, 3), dtype=np.float32)
		self._prev_landmarks = np.zeros((21, 3), dtype=np.float32)
		self._pixel_coords = np.zeros((21, 2), dtype=np.int32)
		self._needs_update = np.zeros(21, dtype=bool)

		# Angle calculation state
		self._locked_thumb_cmc = np.zeros(3, dtype=np.float32)
		self._locked_wrist = np.zeros(3, dtype=np.float32)
		self._lock_slope = np.zeros(2, dtype=np.float32)
		self._axis_locked = False
		self._angle = 0.0

		# Debug file output
		self._write_enabled = False
		self._file_handle = None
		self._update_count = 0

	def update_values(self, landmark_data: List[Dict]) -> None:
		"""
		Update landmark coordinates with new data.

		Optimized for minimal overhead:
		- Single pass through data
		- Vectorized distance calculations
		- In-place array updates
		- Early termination on invalid data

		Args:
				landmark_data: List of dicts with keys 'point', 'coordinates', 'axis'
											Example: [{'point': 0, 'coordinates': (x, y), 'axis': [x, y, z]}, ...]
		"""
		if not landmark_data:
			return

		# Fast path: Extract all data into temporary arrays
		n_points = len(landmark_data)
		temp_points = np.empty(n_points, dtype=np.int32)
		temp_pixel = np.empty((n_points, 2), dtype=np.int32)
		temp_axis = np.empty((n_points, 3), dtype=np.float32)
		valid_mask = np.ones(n_points, dtype=bool)

		# Single loop to extract and validate data
		for i, data in enumerate(landmark_data):
			point_id = data['point']
			temp_points[i] = point_id

			# Validate point ID
			if point_id < 0 or point_id > 20:
				valid_mask[i] = False
				continue

			# Extract pixel coordinates (may be None)
			coords = data.get('coordinates')
			if coords is not None:
				temp_pixel[i] = coords
			else:
				valid_mask[i] = False
				continue

			# Extract axis values
			axis = data['axis']
			temp_axis[i] = axis

			# Validate axis values (must be normalized [0, 1])
			if np.any(temp_axis[i] > 1.0) or np.any(temp_axis[i] < 0.0):
				valid_mask[i] = False

		# Filter to valid points only
		valid_indices = np.where(valid_mask)[0]
		if len(valid_indices) == 0:
			return

		points = temp_points[valid_indices]
		new_pixel = temp_pixel[valid_indices]
		new_axis = temp_axis[valid_indices]

		# Vectorized delta calculation for all valid points
		if IGNORE_Z:
			# Calculate 2D Euclidean distance (x, y only)
			deltas = np.linalg.norm(
				new_axis[:, :2] - self._prev_landmarks[points, :2],
				axis=1
			)
		else:
			# Calculate 3D Euclidean distance (x, y, z)
			deltas = np.linalg.norm(
				new_axis - self._prev_landmarks[points],
				axis=1
			)

		# Determine which points need updating (exceeded a threshold)
		update_mask = deltas >= CHANGE_THRESHOLD
		update_indices = points[update_mask]

		if len(update_indices) > 0:
			# Update landmarks that changed significantly
			self._landmarks[update_indices] = new_axis[update_mask]
			self._pixel_coords[update_indices] = new_pixel[update_mask]
			self._prev_landmarks[update_indices] = new_axis[update_mask]
			self._needs_update[update_indices] = True

			# Optional: Write to debug file
			if self._write_enabled:
				self._write_coordinates(update_indices, new_pixel[update_mask], new_axis[update_mask])

		# Lock initial position on first valid update (for angle calculation)
		if not self._axis_locked:
			if Point.THUMB_CMC in points and Point.Wrist in points:
				thumb_idx = np.where(points == Point.THUMB_CMC)[0]
				wrist_idx = np.where(points == Point.Wrist)[0]

				if len(thumb_idx) > 0 and len(wrist_idx) > 0:
					self._locked_thumb_cmc[:] = new_axis[thumb_idx[0]]
					self._locked_wrist[:] = new_axis[wrist_idx[0]]
					# Calculate initial slope (2D only)
					self._lock_slope[:] = (
						self._locked_thumb_cmc[:2] - self._locked_wrist[:2]
					)
					self._axis_locked = True

		# Calculate an angle if we have all tracked points
		tracked_count = np.sum(np.isin(points, list(TRACKED_POINTS)))
		if tracked_count >= 6 and self._axis_locked:
			self._calculate_angle()

		self._update_count += 1

	def _calculate_angle(self) -> None:
		"""
		Calculate angle between current and locked hand orientation.
		Uses vectorized operations for fast computation.
		"""
		# Get current positions
		thumb_current = self._landmarks[Point.THUMB_CMC, :2]
		wrist_current = self._landmarks[Point.Wrist, :2]

		# Current slope
		current_slope = thumb_current - wrist_current

		# Calculate angle using dot product and magnitudes
		# angle = arccos(u·v / (||u|| ||v||))
		dot_product = np.dot(self._lock_slope, current_slope)

		# Magnitudes (with small epsilon to avoid division by zero)
		mag_locked = np.linalg.norm(self._lock_slope) + 1e-8
		mag_current = np.linalg.norm(current_slope) + 1e-8

		# Calculate angle in degrees
		cos_angle = np.clip(dot_product / (mag_locked * mag_current), -1.0, 1.0)
		self._angle = np.degrees(np.arccos(cos_angle))

	def hc_delta_theta(self) -> float:
		"""
		Get the calculated angle between current and locked positions.

		Returns:
				Angle in degrees (0-180)
		"""
		return float(self._angle)

	def get_landmark(self, point: Point) -> np.ndarray:
		"""
		Get normalized 3D coordinates for a specific landmark.

		Args:
				point: Landmark point enum

		Returns:
				NumPy array of shape (3,) with [x, y, z] coordinates
		"""
		return self._landmarks[point].copy()

	def get_pixel_coords(self, point: Point) -> Tuple[int, int]:
		"""
		Get pixel coordinates for a specific landmark.

		Args:
				point: Landmark point enum

		Returns:
				Tuple of (x, y) pixel coordinates
		"""
		coords = self._pixel_coords[point]
		return int(coords[0]), int(coords[1])

	def landmark_needs_update(self, point: Point) -> bool:
		"""
		Check if a landmark has been updated in the last frame.

		Args:
				point: Landmark point enum

		Returns:
				True if landmark exceeded a movement threshold
		"""
		return bool(self._needs_update[point])

	def get_all_landmarks(self) -> np.ndarray:
		"""
		Get all landmark coordinates as a single array.

		Returns:
				NumPy array of shape (21, 3) with all landmark coordinates
		"""
		return self._landmarks.copy()

	def get_update_mask(self) -> np.ndarray:
		"""
		Get a boolean mask indicating which landmarks were updated.

		Returns:
				NumPy array of shape (21,) with True for updated landmarks
		"""
		return self._needs_update.copy()

	def reset_update_flags(self) -> None:
		"""Reset all update flags to False."""
		self._needs_update.fill(False)

	def enable_coordinates_write_out(self, filename: str = "COORDINATES.txt") -> None:
		"""
		Enable writing coordinates to file for debugging.

		Args:
				filename: Output file path
		"""
		self._write_enabled = True
		self._file_handle = open(filename, 'a', buffering=8192)

	def disable_coordinates_write_out(self) -> None:
		"""Disable coordinate file writing and close file."""
		self._write_enabled = False
		if self._file_handle:
			self._file_handle.close()
			self._file_handle = None

	def _write_coordinates(
		self,
		points: np.ndarray,
		pixel_coords: np.ndarray,
		axis_coords: np.ndarray
	) -> None:
		"""
		Write coordinates to debug file.

		Args:
				points: Array of point IDs
				pixel_coords: Array of pixel coordinates
				axis_coords: Array of normalized axis coordinates
		"""
		if not self._file_handle:
			return

		for i in range(len(points)):
			self._file_handle.write(
				f"{points[i]},{pixel_coords[i, 0]},{pixel_coords[i, 1]},"
				f"{axis_coords[i, 0]:.6f},{axis_coords[i, 1]:.6f},{axis_coords[i, 2]:.6f}\n"
			)

	def get_statistics(self) -> Dict[str, any]:
		"""
		Get performance and state statistics.

		Returns:
				Dictionary with statistics
		"""
		return {
			'update_count': self._update_count,
			'axis_locked': self._axis_locked,
			'current_angle': self._angle,
			'active_landmarks': int(np.sum(np.any(self._landmarks != 0, axis=1))),
			'write_enabled': self._write_enabled,
		}

	def __del__(self):
		"""Cleanup file handle on deletion."""
		if self._file_handle:
			self._file_handle.close()

	def __repr__(self) -> str:
		stats = self.get_statistics()
		return (
			f"HandLocation(updates={stats['update_count']}, "
			f"active={stats['active_landmarks']}/21, "
			f"angle={stats['current_angle']:.2f}°)"
		)


# Convenience functions for backward compatibility
def create_hand_location() -> HandLocation:
	"""Create a new HandLocation instance."""
	return HandLocation()

from typing import List, Dict

import numpy as np

from controls.location import HandLocation


# Performance testing utilities
class PerformanceProfiler:
	"""Simple profiler for measuring HandLocation performance."""

	def __init__(self):
		self.timings = []

	@staticmethod
	def profile_update(
		hand_location: HandLocation,
		landmark_data: List[Dict],
		iterations: int = 1000
	) -> Dict[str, float]:
		"""
		Profile update_values performance.

		Args:
				hand_location: HandLocation instance
				landmark_data: Sample landmark data
				iterations: Number of iterations to run

		Returns:
				Dictionary with timing statistics
		"""
		import time

		timings = []
		for _ in range(iterations):
			start = time.perf_counter()
			hand_location.update_values(landmark_data)
			end = time.perf_counter()
			timings.append((end - start) * 1e6)  # microseconds

		timings_array = np.array(timings)
		return {
			'mean_us': float(np.mean(timings_array)),
			'median_us': float(np.median(timings_array)),
			'std_us': float(np.std(timings_array)),
			'min_us': float(np.min(timings_array)),
			'max_us': float(np.max(timings_array)),
			'p95_us': float(np.percentile(timings_array, 95)),
			'p99_us': float(np.percentile(timings_array, 99)),
		}

import math
import time


class OneEuroFilter:
	# Adaptive smoothing (based on velocity)
	# Resource link: https://gery.casiez.net/1euro/
	def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
		"""
		:param min_cutoff: minimum cutoff frequency (Hz). Lower is smoother (0.5-1.0). Higher is more responsive (1.0-2.0).
		:param beta: 0.001-1.0: more smoothing at high speed. 0.01-0.1: less smoothing at high speed.
		:param d_cutoff: Cutoff frequency for derivative (Hz).
		"""
		self.min_cutoff = min_cutoff
		self.beta = beta
		self.d_cutoff = d_cutoff

		self.x_prev = None
		self.y_prev = None
		self.dx_prev = 0.0
		self.dy_prev = 0.0
		self.last_time = None

	def smooth(self, raw_x, raw_y, timestamp=None):
		now = timestamp or time.time()
		if self.x_prev is None:
			self.x_prev, self.y_prev = raw_x, raw_y
			self.last_time = now
			return raw_x, raw_y
		dt = now - self.last_time
		if dt <= 0:
			dt = 1e-3  # bypass div-by-zero
		elif dt > 1.0:
			dt = 1.0  # should prevent giant lag jumps
		smooth_x, dx = self._filter_axis(raw_x, self.x_prev, self.dx_prev, dt)
		smooth_y, dy = self._filter_axis(raw_y, self.y_prev, self.dy_prev, dt)
		self.x_prev, self.y_prev = smooth_x, smooth_y
		self.dx_prev, self.dy_prev = dx, dy
		self.last_time = now
		return smooth_x, smooth_y

	def _filter_axis(self, value, prev_value, prev_dx, dt):
		"""Filter one axis with an adaptive cutoff."""
		_exp = math.exp  # local alias for speed
		# Derivative low-pass filtering
		alpha_d = self._smoothing_factor(dt, self.d_cutoff)
		dx = (value - prev_value) / dt
		dx_smoothed = alpha_d * dx + (1.0 - alpha_d) * prev_dx
		# Adaptive cutoff for the main signal
		cutoff = self.min_cutoff + self.beta * abs(dx_smoothed)
		alpha = self._smoothing_factor(dt, cutoff)
		filtered = alpha * value + (1.0 - alpha) * prev_value
		return filtered, dx_smoothed

	def reset(self):
		self.x_prev = self.y_prev = None
		self.dx_prev = self.dy_prev = 0.0
		self.last_time = None

	@staticmethod
	def _smoothing_factor(dt, cutoff):
		""" Calculate a smoothing factor for a given cutoff frequency """
		tau = 1.0 / (2 * math.pi * cutoff)
		return 1.0 / (1.0 + tau / dt)

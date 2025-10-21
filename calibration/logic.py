
from controls.tracking import HandTracker


# What algorithm to use...
# The current calibration algorithm is set to use 5-point calibration in following pattern:
# +------------------------------------+
# | 1:Top-L                2: Top-R    |
# |           3: Center                |
# | 4:Bottom-L             5: Bottom-R |
# +------------------------------------+
# Can alternatively use the Plus-points as well. However, going to stick with this one due to its simplicity (one can argue the other is simpler too).
# The current approach does not take into account:
# 1. the eventual camera movement
# 2. User is unable to reach all corners
# 3. User moves Position
# LIMITATIONS:
# This calibration limits the usability of this tool to a single monitor only
class Calibration:
	def __init__(self, source: int | None):
		self.points = []
		self.tracker = HandTracker(False, True)

	def calibrate(self):
		# TODO: Pass the logic to capture the calibration points i.e., the tracking points dynamically.
    # Pass the capture function here. Will alter the run function in HandTracker to accept a capture function(s).
		...

#!/bin/env python

from multiprocessing import Process
from tracking.points import Points
from numba import jit
import asyncio
import time

"""
	0: Wrist							1: THUMB_CMC					2: THUMB_IP
	3: THUMB_MCP					4: THUMB_TIP					5: INDEX_FINGER_MCP
	6: INDEX_FINGER_PIP		7: INDEX_FINGER_DIP		8: INDEX_FINGER_TIP
	9: MIDDLE_FINGER_MCP	10: MIDDLE_FINGER_PIP	11: MIDDLE_FINGER_DIP
	12: MIDDLE_FINGER_TIP	13: RING_FINGER_MCP		14: RING_FINGER_PIP
	15: RING_FINGER_DIP		14: RING_FINGER_TIP		17: PINKY_MCP
	18: PINKY_PIP					19: PINKY_DIP					20: PINKY_TIP
"""


class Coordinates:
		"""
		@pnt: for debugging ...
		"""
		def __init__(self, pnt=None):
				self.point = pnt if pnt is not None else None
				self.coordinates = []
				self.x_cord = None
				self.y_cord = None
				self.z_cord = None

		def update_value(self, coordinates, x_cord, y_cord, z_cord):
				self.coordinates = coordinates
				self.x_cord = x_cord
				self.y_cord = y_cord
				self.z_cord = z_cord

		def values(self):
				return self.coordinates, self.x_cord, self.y_cord, self.z_cord


class HandLocation:
		def __set_default_values(self):
				self.coords[Points.Wrist] = Coordinates(Points.Wrist)
				self.coords[Points.THUMB_CMC] = Coordinates(Points.THUMB_CMC)
				self.coords[Points.THUMB_IP] = Coordinates(Points.THUMB_IP)
				self.coords[Points.THUMB_MCP] = Coordinates(Points.THUMB_MCP)
				self.coords[Points.THUMB_TIP] = Coordinates(Points.THUMB_TIP)
				self.coords[Points.INDEX_FINGER_MCP] = Coordinates(Points.INDEX_FINGER_MCP)
				self.coords[Points.INDEX_FINGER_PIP] = Coordinates(Points.INDEX_FINGER_PIP)
				self.coords[Points.INDEX_FINGER_DIP] = Coordinates(Points.INDEX_FINGER_DIP)
				self.coords[Points.INDEX_FINGER_TIP] = Coordinates(Points.INDEX_FINGER_TIP)
				self.coords[Points.MIDDLE_FINGER_MCP] = Coordinates(Points.MIDDLE_FINGER_MCP)
				self.coords[Points.MIDDLE_FINGER_PIP] = Coordinates(Points.MIDDLE_FINGER_PIP)
				self.coords[Points.MIDDLE_FINGER_DIP] = Coordinates(Points.MIDDLE_FINGER_DIP)
				self.coords[Points.MIDDLE_FINGER_TIP] = Coordinates(Points.MIDDLE_FINGER_TIP)
				self.coords[Points.RING_FINGER_MCP] = Coordinates(Points.RING_FINGER_MCP)
				self.coords[Points.RING_FINGER_PIP] = Coordinates(Points.RING_FINGER_PIP)
				self.coords[Points.RING_FINGER_DIP] = Coordinates(Points.RING_FINGER_DIP)
				self.coords[Points.RING_FINGER_TIP] = Coordinates(Points.RING_FINGER_TIP)
				self.coords[Points.PINKY_MCP] = Coordinates(Points.PINKY_MCP)
				self.coords[Points.PINKY_PIP] = Coordinates(Points.PINKY_PIP)
				self.coords[Points.PINKY_DIP] = Coordinates(Points.PINKY_DIP)
				self.coords[Points.PINKY_TIP] = Coordinates(Points.PINKY_TIP)


		"""
		@__launch_async_proc:
		- Kinda bad idea but I'll work with it for now ...
		!TODO:
		- Other than that, pass this value to a C++ function that will handle all the processing
		- shenanigans
		"""
		def __launch_async_proc(self):
				while self.event_control:
					time.sleep(0.1)
					if self.update is not None:
							cord, x, y, z = self.coords[self.update].values()
							print(self.coords[self.update].values())
					break

		def __init__(self):
				self.coords = dict()
				self.__set_default_values()
				self.event_control = False
				self.update = None

		def update_value(self, point, coordinates, x_cord, y_cord, z_cord):
				self.update = point
				values = {
						'coordinates': coordinates,
						'x_cord': x_cord,
						'y_cord': y_cord,
						'z_cord': z_cord,
				}
				self.coords[point].update_value(**values)

		# !TODO: send a signal to stop the event loop
		def take_action(self):
				if self.event_control == False:
						self.event_control = True
						proc = Process(target=self.__launch_async_proc)
						proc.start()
				# Attach it


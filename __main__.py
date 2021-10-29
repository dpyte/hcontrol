#!/bin/env python

USE_BUILD_HC = True

import cv2
import mediapipe as mp
if USE_BUILD_HC:
	from build.tracking.exec import HandCoordinates as HC


DEBUG = True


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
location = HC.HandLocation()


"""
 Idea is to having an asynchronous thread that reads off the data from the
 init function and acts based on the collected data

 @take_async_action:
 in HandLocation can be used to trace and create a singular output
 based on the data collected over the period of n milliseconds

"""

"""
	0: Wrist							1: THUMB_CMC					2: THUMB_IP
	3: THUMB_MCP					4: THUMB_TIP					5: INDEX_FINGER_MCP
	6: INDEX_FINGER_PIP		7: INDEX_FINGER_DIP		8: INDEX_FINGER_TIP
	9: MIDDLE_FINGER_MCP	10: MIDDLE_FINGER_PIP	11: MIDDLE_FINGER_DIP
	12: MIDDLE_FINGER_TIP	13: RING_FINGER_MCP		14: RING_FINGER_PIP
	15: RING_FINGER_DIP		14: RING_FINGER_TIP		17: PINKY_MCP
	18: PINKY_PIP					19: PINKY_DIP					20: PINKY_TIP
"""

location.take_action()

def extract_coordinates_from_hand_landmark(hands, lm, i_width, i_height):
		for point in hands.HandLandmark:
				normalize = lm.landmark[point]
				coordinated = mp_drawing._normalized_to_pixel_coordinates(
						normalize.x,
						normalize.y,
						i_width,
						i_height
				)
				if DEBUG:
						print('Point: {}\nCoordinates: {}\nLandmarks:\n{}'.format(
								point,
								coordinated,
								normalize)
						)
				# X -> Left & Right
				# Y -> Down & Up
				# Z -> Back & foreword
				values = {
						"point": int(point),
						"coordinates": (coordinated),
						"axis": [normalize.x, normalize.y, normalize.z]
				}
				location.update_values(values)


def init():
		cap = cv2.VideoCapture(0)
		with mp_hands.Hands(
				min_detection_confidence=0.9,
				min_tracking_confidence=0.5
		) as hands:
				while cap.isOpened():
						succ, img = cap.read()
						if not succ:
								print('ignoring empty camera frame')
								continue
						img.flags.writeable = False
						img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
						results = hands.process(img)

						height, width, _ = img.shape
						img.flags.writeable = True
						img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
						if results.multi_hand_landmarks:
								for hl in results.multi_hand_landmarks:
										extract_coordinates_from_hand_landmark(
												mp_hands,
												hl,
												width,
												height
										)
										mp_drawing.draw_landmarks(
												img,
												hl,
												mp_hands.HAND_CONNECTIONS,
												mp_drawing_styles.get_default_hand_landmarks_style(),
												mp_drawing_styles.get_default_hand_connections_style()
										)
						if DEBUG:
								cv2.imshow('HC Hands', cv2.flip(img, 1))
						if cv2.waitKey(5) & 0xFF == 27:
								break
		cap.release()


if __name__ == '__main__':
		init()

#!/bin/env python
# -*- coding: utf-8 -*-

import glob
import os

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as pyplt
from PIL import Image

mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

CAPTURE_IMG = False
USE_FRAME_CAPTURE = False

flloc = os.path.dirname(os.path.abspath(__file__))


def load_images():
		IMG_DIR = './img/'
		img_files = []
		[img_files.extend(glob.glob(IMG_DIR + '*' + x)) for x in 'png']
		return img_files


def load_df():
		DF_DIR = './jupyter/pixel_values/'
		df_files = []
		[df_files.extend(glob.glob(DF_DIR + '*' + x)) for x in 'csv']
		return df_files


def extract_coordinates_from_hand_landmark(hands, lm, i_width, i_height):
		data = []
		for point in hands.HandLandmark:
				normalize = lm.landmark[point]
				coordinated = mp_drawing._normalized_to_pixel_coordinates(
						normalize.x,
						normalize.y,
						i_width,
						i_height
				)
				values = {
						"point": int(point),
						"coordinates": coordinated,
						"axis": [normalize.x, normalize.y, normalize.z]
				}
				data.append(values)
		return data


def process_from_big_data():
		dfs = load_df()
		for i in dfs:
				df = pd.read_csv(i, header=None, sep=',')
				df = df.iloc[1:, 1:]
				pyplt.plot(df)
				print(df)


def roi(frame, vert):
		"""
				@frame: reference frame in PIL format
				@vert: vertices of the image ...
		"""
		print('Region of interest: [{}, {}]  [{}, {}]  [{}, {}]  [{}, {}]'.format(
				vert[0][0], vert[0][1],
				vert[1][0], vert[1][1],
				vert[2][0], vert[2][1],
				vert[3][0], vert[3][1]
		))
		area = (vert[0][0] - 8, vert[0][1] + 10, vert[3][0] + 6, vert[3][1] + 6)
		cropped = frame.crop(area)
		return cropped


def capture_frame_window(axis):
		"""
				Capture frame of interest based on the chosen furthest coordinates
		"""
		p1 = list(filter(lambda ax: ax['point'] == 20, axis))[0]['coordinates']
		p2 = list(filter(lambda ax: ax['point'] == 16, axis))[0]['coordinates']
		p3 = list(filter(lambda ax: ax['point'] == 8, axis))[0]['coordinates']
		p4 = list(filter(lambda ax: ax['point'] == 4, axis))[0]['coordinates']
		return [p2[0], p3[1]], [p4[0], p3[1]], \
		       [p2[0], p1[1]], [p4[0], p1[1]]


def process_img(screen, data):
		"""
				Extract coordinates and pass into the Region-Of-Interest function
		"""
		p1, p2, p3, p4 = capture_frame_window(data)
		vertices = np.array([p1, p2, p3, p4], np.int32)
		ns = roi(screen, vertices)
		return ns


def extract_roi_from_image():
		"""
				Run MediaPipe scan on extracted images and save it using those
				extracted coordinates
		"""
		for idx, file in enumerate(load_images()):
				with mp_hands.Hands(
						static_image_mode=True,
						min_detection_confidence=0.9,
						min_tracking_confidence=0.5
				) as hands:
						# for idx, file in enumerate(load_images()):
						image = cv2.flip(cv2.imread(file), 1)
						frame = Image.fromarray(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
						results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
						ns = None
						if results.multi_hand_landmarks:
								img_h, img_w, _ = image.shape
								for hl in results.multi_hand_landmarks:
										data = extract_coordinates_from_hand_landmark(
												mp_hands,
												hl,
												img_w,
												img_h
										)
										ns = process_img(frame, data)
						if ns is not None:
								cv2.imshow('New Buffer', cv2.flip(ns, 1))
						else:
								cv2.imshow('MP Hands', cv2.flip(image, 1))


def extract_roi_from_live():
		SAVE_LOCATION = os.path.join(flloc, 'img/')
		image_counter = 0
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

						frame = Image.fromarray(img)
						results = hands.process(img)

						height, width, _ = img.shape
						img.flags.writeable = True
						img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
						if results.multi_hand_landmarks:
								for hl in results.multi_hand_landmarks:
										data = extract_coordinates_from_hand_landmark(
												mp_hands, hl,
												width, height)
										mp_drawing.draw_landmarks(
												img, hl,
												mp_hands.HAND_CONNECTIONS,
												mp_drawing_styles.get_default_hand_landmarks_style(),
												mp_drawing_styles.get_default_hand_connections_style()
										)
								"""
										Process image and save it using the specified coordinates
								"""
								save_location = os.path.join(SAVE_LOCATION, f'image_{image_counter}.png')
								ns = process_img(frame, data)
								ns.save(save_location, format='png')
								image_counter += 1
						cv2.imshow('HC Hands', cv2.flip(img, 1))
						if cv2.waitKey(5) & 0xFF == 27:
								break
		cap.release()


def init():
		if USE_FRAME_CAPTURE:
				process_from_big_data()
		else:
				"""
					Dump rest of the stuff in here
				"""
				extract_roi_from_live()


if __name__ == '__main__':
		init()

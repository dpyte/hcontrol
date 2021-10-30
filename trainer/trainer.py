#!/bin/env python


import numpy as np
import pandas as pd


def file_path():
		train = 'data/SPANNING_COORDINATES.csv'
		return train


def break_down(df):
		"""
		Throw away entire row with its Y and Z value equal to 0
		"""
		new_df = df.drop(df.index[(x for x, y in df.iterrows() if y['Y_Axis'] == 0 or y['Z_Axis'] == 0)])
		return new_df


def main():
		train_data = file_path()
		train = pd.read_csv(train_data, sep=',')
		train = break_down(train)
		print(train)


if __name__ == '__main__':
		main()

#!/bin/env python

import mpld3
import plotly
import numpy as np
import pandas as pd
import plotly.graph_objs as go

plotly.offline.init_notebook_mode()

NAME_EXTENSION = '_COORDINATES.txt'


def dataframe(name):
		df = pd.read_csv(name, header=None, sep=',')
		ini = 2
		rng_1 = np.arange(0, len(df), 3)
		rng_2 = np.arange(1, len(df), 3)
		df.drop(labels=rng_1, axis=0, inplace=True)
		df.drop(labels=rng_2, axis=0, inplace=True)
		return df


def graph(df, name):
	trace = go.Scatter3d(
		x=df[3],
		y=df[4],
		z=df[5],
		mode='markers',
		marker={
			'size': 2,
			'opacity': 0.01,
		}
	)
	layout = go.Layout(
		margin={
			'l': 0,
			'r': 0,
			'b': 0,
			't': 0
		}
	)
	data = [trace]
	plt_fig = go.Figure(data=data, layout=layout)
	plotly.offline.iplot(plt_fig)


def init():
		spdf = dataframe(f'./data/SPANNING{NAME_EXTENSION}')
		nwdf = dataframe(f'./data/NORMAL_WAIVE{NAME_EXTENSION}')
		itdf = dataframe(f'./data/INDEX_TAP{NAME_EXTENSION}')
		fwdf = dataframe(f'./data/FIRST_WAIVE{NAME_EXTENSION}')
		bfdf = dataframe(f'./data/BACK_FORTH{NAME_EXTENSION}')

		graph(spdf, 'SPANNING')
		graph(nwdf, 'NORMAL_WAIVE')
		graph(itdf, "INDEX_TAP")
		graph(fwdf, "FIRST_WAIVE")
		graph(bfdf, "BACK_FORTH")


if __name__ == '__main__':
		init()

#!/bin/env python

import os
import platform
import sysconfig
import sys
import subprocess

prog_name = 'hcontrol'
ccflags = [
	"-mmcu={MCU}"
	]

cxxflags = []
cpppath = []

libpath = [
	"/usr/local/lib",
	"/usr/lib",
	]

env = Environment(
	ENV = {
		'PATH': os.environ['PATH'],
	},
	CCFLAGS=[
		"-g" if ARGUMENTS.get('debug', 0) else "-O2",
		"-fPIC",
		"-O2",
		"-Wall",
		"-Wno-main",
		"-Wundef",
		"-Wstrict-prototypes",
	] + ccflags,
	CC='avr-gcc'
	)


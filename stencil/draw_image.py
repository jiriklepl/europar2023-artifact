#!/usr/bin/env python

import pandas as pd
import numpy as np
from PIL import Image
import sys
import argparse

parser = argparse.ArgumentParser(
	prog='gen-image',
	description='Generate an image represented as a sequence of (HEIGHT * WIDTH * CHANNELS) floating-point numbers in [0; 256)',
	epilog='output: csv list of points which should be clustered',
	add_help=True,
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('width', type=int, help='the with of the image')
parser.add_argument('height', type=int, help='the height of the image')
parser.add_argument('channels', type=int, help='the number of color channels')

if __name__=="__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)

	image = pd.read_csv(sys.stdin, header=None).to_numpy().astype(np.uint8)

	Image.fromarray(image.reshape((args.height, args.width, args.channels))).show()

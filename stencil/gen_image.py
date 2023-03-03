#!/usr/bin/env python

import numpy as np
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

	numbers = np.zeros(shape=(args.height, args.width, args.channels), dtype=np.uint8)

	rect = numbers[args.height//3 : 2 * args.height//3, args.width//3 : 2 * args.width//3]
	rect[:,:,:] = np.random.randint(low=0, high=256, size=rect.size, dtype=np.uint8).reshape(rect.shape)

	[print(n) for n in numbers.reshape((-1))]

#!/usr/bin/env python

import pandas as pd
import numpy as np
import sklearn.cluster
import argparse
import sys
import time

parser = argparse.ArgumentParser(
	prog='kmeans-triv',
	description='Performs kmeans algorithm with a trivial configuration (see the code)',
	add_help=True,
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('size', type=int, help='the number of points')
parser.add_argument('clusters', type=int, help='the number of clusters')
parser.add_argument('dimensions', type=int, default=4, help='the number of dimensions of the space')

if __name__=="__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)

	data = np.array(pd.read_csv(sys.stdin, header=None, delimiter=' '))

	kmeans = sklearn.cluster.KMeans(n_clusters=args.clusters, init=data[0:args.clusters], n_init=1, max_iter=10, tol=0., copy_x=False)

	start = time.time_ns()
	kmeans.fit(data)
	end = time.time_ns()

	print(str(end-start), file=sys.stderr)

	print('== MEANS ==')




	for c in kmeans.cluster_centers_:
		for d in c:
			print(f"{np.double(d):.0f}")

	print('== ASSIGNMENTS ==')

	[print(c) for c in kmeans.predict(data)]

	# for c in

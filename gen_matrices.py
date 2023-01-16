#!/usr/bin/python3

import random, struct, sys

len(sys.argv) == 2 or exit(f'Usage: {sys.argv[0]} OUTPUT_FILE')

with open(sys.argv[1], 'wb') as out:
	n = 2*8192
	for i in range(n):
		out.write(b''.join(random.choices([struct.pack('<f', f) for f in range(4)], k=8192)))
		print(i*10000//n/100, end='%\r')
